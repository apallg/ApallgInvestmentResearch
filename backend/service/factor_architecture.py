import datetime as dt
import hashlib
import json
import os
import random
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.service.factor_panel import FactorPanelService
from backend.service.factor_rolling import FactorRollingService
from backend.service.qlib_access import run_with_qlib_read_access
from core.adapters.qlib.factor_resolver import resolve_factor_expression
from core.data import Database
from core.data.repos.factor_architecture import factor_architecture_repo
from core.domain.factors.evidence import (
    orient_residual_rows_walkforward,
    summarize_factor_evidence,
    summarize_ic_values,
)
from core.domain.factors.preprocessing import preprocess_candidates
from core.domain.factors.rolling import _add_months
from core.domain.factors.role_selection import (
    build_balanced_set,
    build_balanced_extension,
    select_core_candidates,
    select_satellite_candidates,
)
from core.domain.factors.validation import OverfittingChecker
from core.domain.factors.walkforward import evenly_sample_dates, purge_dates_before_boundary
from core.shared.infrastructure.async_task import task_manager
from core.shared.infrastructure.async_task import TaskCancelled


def _fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_FACTOR_SET_VALIDATION_POLICY_VERSION = "factor_set_validation_attestation_v2"


def _validation_policy(role, scope, config):
    policy = {
        "version": _FACTOR_SET_VALIDATION_POLICY_VERSION,
        "role": role,
        "scope": scope,
        "min_folds": max(3, int(config.get("min_folds", 3))),
    }
    if role == "core" and scope == "fixed_oos":
        policy.update({
            "min_positive_ratio": max(0.6, float(config.get("oos_positive_ratio", 0.6))),
            "min_random_percentile": max(0.75, float(config.get("min_random_percentile", 0.75))),
            "min_random_count": max(10, int(config.get("min_random_count", 10))),
        })
    elif role == "core":
        policy.update({
            "min_positive_ratio": max(0.5, float(config.get("ab_positive_ratio", 0.5))),
            "min_random_percentile": max(0.75, float(config.get("min_random_percentile", 0.75))),
            "min_random_count": max(10, int(config.get("min_random_count", 10))),
        })
    else:
        policy.update({
            "min_positive_ratio": max(0.6, float(config.get("satellite_ab_positive_ratio", 0.6))),
            "min_median_delta": max(0.0, float(config.get("satellite_min_delta", 0.0))),
            "min_unconditional_delta": max(-0.005, float(config.get("satellite_ab_unconditional_floor", -0.005))),
        })
    return policy


def _build_validation_evidence(run_id, factor_set, role, scope, analysis, config, results,
                               evaluation_set_id=None):
    policy = _validation_policy(role, scope, config)
    folds_by_arm = {
        arm: sorted({str(fold.get("fold_id")) for fold in (result.get("folds") or [])
                     if fold.get("fold_id") is not None})
        for arm, result in results.items()
    }
    arm_fingerprints = {
        arm: _fingerprint({
            "factor_keys": result.get("factor_keys") or [],
            "summary": {key: value for key, value in result.items() if key != "folds"},
            "folds": sorted(result.get("folds") or [], key=lambda fold: str(fold.get("fold_id") or "")),
        })
        for arm, result in results.items()
    }
    analysis_snapshot = json.loads(json.dumps(analysis, ensure_ascii=False, default=str))
    return {
        "version": _FACTOR_SET_VALIDATION_POLICY_VERSION,
        "run_id": run_id,
        "set_id": factor_set["set_id"],
        "evaluation_set_id": evaluation_set_id or factor_set["set_id"],
        "set_fingerprint": factor_set.get("fingerprint"),
        "factor_keys_fingerprint": _fingerprint(factor_set.get("factor_keys") or []),
        "role": role,
        "scope": scope,
        "validated": analysis_snapshot.get("validated") is True,
        "policy": policy,
        "policy_fingerprint": _fingerprint(policy),
        "analysis": analysis_snapshot,
        "analysis_fingerprint": _fingerprint(analysis_snapshot),
        "folds_by_arm": folds_by_arm,
        "arm_fingerprints": arm_fingerprints,
    }


def _has_current_validation_evidence(factor_set, run_id, role, scope):
    metrics = factor_set.get("metrics") or {}
    evidence = metrics.get("validation_evidence")
    if not isinstance(evidence, dict):
        return False
    policy = evidence.get("policy") or {}
    analysis = evidence.get("analysis") or {}
    if not isinstance(policy, dict) or not isinstance(analysis, dict):
        return False
    fold_count = analysis.get("fold_count")
    winner = {}
    if role == "core" and scope == "fixed_oos":
        winner = (analysis.get("candidate_results") or {}).get(analysis.get("winner_arm")) or {}
        fold_count = winner.get("fold_count")
    policy_valid = (
        policy.get("version") == _FACTOR_SET_VALIDATION_POLICY_VERSION
        and policy.get("role") == role
        and policy.get("scope") == scope
        and evidence.get("policy_fingerprint") == _fingerprint(policy)
    )
    if not policy_valid:
        return False
    try:
        if role == "core" and scope == "fixed_oos":
            thresholds_valid = (
                winner.get("eligible") is True
                and float(winner.get("rank_ic") or 0) > 0
                and int(winner.get("fold_count") or 0) >= int(policy.get("min_folds") or 1)
                and float(winner.get("positive_fold_ratio") or 0)
                    >= float(policy.get("min_positive_ratio") or 0)
                and int(analysis.get("random_count") or 0) >= int(policy.get("min_random_count") or 0)
                and winner.get("random_percentile") is not None
                and float(winner["random_percentile"]) >= float(policy.get("min_random_percentile") or 0)
            )
        elif role == "core" and scope == "development":
            families = analysis.get("family_results") or {}
            thresholds_valid = (
                int(analysis.get("fold_count") or 0) >= int(policy.get("min_folds") or 1)
                and float(analysis.get("full_rank_ic") or 0) > 0
                and int(analysis.get("random_count") or 0) >= int(policy.get("min_random_count") or 0)
                and analysis.get("random_percentile") is not None
                and float(analysis["random_percentile"]) >= float(policy.get("min_random_percentile") or 0)
                and bool(families)
                and not analysis.get("harmful_families")
                and all(
                    int(item.get("fold_count") or 0) >= int(policy.get("min_folds") or 1)
                    and item.get("median_delta") is not None
                    and float(item["median_delta"]) > 0
                    and float(item.get("positive_ratio") or 0) >= float(policy.get("min_positive_ratio") or 0)
                    and item.get("keep") is True
                    for item in families.values()
                )
            )
        elif role in ("trend", "rebound") and scope == "state_conditional":
            unconditional = analysis.get("unconditional") or {}
            thresholds_valid = (
                analysis.get("role") == role
                and int(analysis.get("fold_count") or 0) >= int(policy.get("min_folds") or 1)
                and analysis.get("median_delta") is not None
                and float(analysis["median_delta"]) > float(policy.get("min_median_delta") or 0)
                and float(analysis.get("positive_ratio") or 0) >= float(policy.get("min_positive_ratio") or 0)
                and unconditional.get("median_delta") is not None
                and float(unconditional["median_delta"]) > float(policy.get("min_unconditional_delta") or 0)
            )
        else:
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        factor_set.get("status") == "validated"
        and evidence.get("version") == _FACTOR_SET_VALIDATION_POLICY_VERSION
        and evidence.get("run_id") == run_id
        and evidence.get("set_id") == factor_set.get("set_id")
        and evidence.get("set_fingerprint")
        and evidence.get("set_fingerprint") == factor_set.get("fingerprint")
        and evidence.get("factor_keys_fingerprint") == _fingerprint(factor_set.get("factor_keys") or [])
        and evidence.get("role") == role
        and evidence.get("scope") == scope
        and evidence.get("validated") is True
        and analysis.get("validated") is True
        and analysis.get("scope") == scope
        and evidence.get("analysis_fingerprint") == _fingerprint(analysis)
        and policy_valid
        and thresholds_valid
        and int(fold_count or 0) >= int(policy.get("min_folds") or 1)
        and (scope != "fixed_oos" or bool(winner.get("eligible")))
        and bool(evidence.get("folds_by_arm"))
    )


class FactorArchitectureService:
    def create(self, request):
        config, variants, preprocessing = self._normalize(request)
        eligible = {item["factor_key"] for item in preprocessing["eligible"]}
        selected_variants = [
            variant for variant in variants
            if (variant.get("variant_key") or variant.get("template_key")) in eligible
        ]
        if not selected_variants:
            raise ValueError("候选池预处理后没有可评估因子")
        run_id = uuid.uuid4().hex[:12]
        factor_architecture_repo.ensure_tables()
        pool_hash = _fingerprint([
            (item["factor_key"], item["expression_fingerprint"])
            for item in preprocessing["eligible"]
        ])
        factor_architecture_repo.create_run(
            run_id, config, pool_hash, config["selection_cutoff"],
        )
        factor_architecture_repo.save_candidates(
            run_id, preprocessing["eligible"], preprocessing["rejected"],
        )
        task_manager.submit_with_id(
            run_id,
            lambda: run_with_qlib_read_access(self._run, run_id, config, selected_variants, preprocessing),
            f"核心基底滚动证据 {len(selected_variants)}因子",
            pool="default",
        )
        return {
            "run_id": run_id,
            "status": "pending",
            "eligible_count": len(selected_variants),
            "rejected_count": preprocessing["rejected_count"],
            "config": config,
        }

    def _normalize(self, request):
        start = str(request.get("start") or "")[:10]
        end = str(request.get("end") or "")[:10]
        if not start or not end or dt.date.fromisoformat(start) >= dt.date.fromisoformat(end):
            raise ValueError("start/end 必须是递增的有效日期")
        factor_source = str(request.get("factor_source") or "pool")
        if factor_source not in ("pool", "public"):
            raise ValueError("factor_source 仅支持 pool/public")
        label_horizon = int(request.get("label_horizon", 10))
        if label_horizon < 2:
            raise ValueError("label_horizon 至少为2")
        label_mode = request.get("label_mode", "tradable_open_to_open")
        if label_mode != "tradable_open_to_open":
            raise ValueError("label_mode 必须为 tradable_open_to_open")
        selection_cutoff = str(request.get("selection_cutoff") or dt.date.today())[:10]
        if dt.date.fromisoformat(selection_cutoff) > dt.date.today():
            raise ValueError("selection_cutoff 不能晚于今天")
        for key, minimum, maximum, default in (
            ("step_months", 1, 12, 3), ("train_months", 12, 120, 60),
            ("valid_months", 1, 24, 6), ("test_months", 1, 12, 3),
            ("purge_gap", label_horizon, 120, 21),
            ("max_eval_dates", 12, 60, 24),
        ):
            value = int(request.get(key, default))
            if value < minimum or value > maximum:
                raise ValueError(f"{key} 必须在 {minimum} 到 {maximum} 之间")
        config = {
            "start": start,
            "end": end,
            "selection_cutoff": selection_cutoff,
            "pool": request.get("pool", "a500"),
            "factor_source": factor_source,
            "source": request.get("source"),
            "factor_keys": list(dict.fromkeys(request.get("factor_keys") or [])),
            "step_months": int(request.get("step_months", 3)),
            "train_months": int(request.get("train_months", 60)),
            "valid_months": int(request.get("valid_months", 6)),
            "test_months": int(request.get("test_months", 3)),
            "purge_gap": int(request.get("purge_gap", 21)),
            "label_horizon": label_horizon,
            "label_mode": label_mode,
            "eval_weeks": int(request.get("eval_weeks", 18)),
            "min_coverage": float(request.get("min_coverage", 0.8)),
            "min_folds": int(request.get("min_folds", 3)),
            "min_positive_ratio": float(request.get("min_positive_ratio", 0.6)),
            "min_state_count": int(request.get("min_state_count", 2)),
            "min_state_floor": float(request.get("min_state_floor", -0.005)),
            "core_max_count": int(request.get("core_max_count", 30)),
            "family_fraction": float(request.get("family_fraction", 0.2)),
            "state_source": "market_regime_history_volatility_axis",
            "max_eval_dates": int(request.get("max_eval_dates", 24)),
            "satellite_max_count": int(request.get("satellite_max_count", 12)),
            "satellite_state_specificity": float(request.get("satellite_state_specificity", 0.02)),
            "satellite_family_overlap_limit": request.get("satellite_family_overlap_limit", 3),
            "satellite_ab_positive_ratio": float(request.get("satellite_ab_positive_ratio", 0.6)),
            "satellite_min_delta": float(request.get("satellite_min_delta", 0.0)),
            "satellite_ab_unconditional_floor": float(request.get("satellite_ab_unconditional_floor", -0.005)),
            "satellite_fusion_weight": float(request.get("satellite_fusion_weight", 0.5)),
            "write_pool": False,
            "profile": "factor_architecture_v1",
        }
        service = FactorRollingService()
        variants = service.resolve_variants(
            config["factor_source"], config["source"], config["factor_keys"],
        )
        from core.domain.factors.registry import get_pool
        from core.domain.factors.selector import get_unified_category
        from core.domain.factors.expression import parse_factor
        active = get_pool().get_active_factors()
        candidates = []
        runtime_variants = []
        checker = OverfittingChecker()
        for source_variant in variants:
            variant = dict(source_variant)
            key = variant.get("variant_key") or variant.get("template_key")
            info = active.get(key, {})
            expression, skip_reason = resolve_factor_expression(key)
            # 回退路径也必须是因果表达式: 解析失败时, 对 raw expr 同样做 WorldQuant→Qlib
            # 方向翻转 + 前视检测, 不允许把未翻转的未来函数表达式带入筛选。
            if not expression:
                fallback = variant.get("expr") or info.get("expr")
                if fallback and not str(fallback).startswith("#"):
                    from core.adapters.qlib.custom_handler import _fix_qlib_compat
                    from core.adapters.qlib.factor_resolver import detect_future_shift
                    fallback = _fix_qlib_compat(str(fallback))
                    if str(fallback).startswith("#"):
                        skip_reason = fallback
                    elif detect_future_shift(fallback):
                        skip_reason = "# SKIP: future-shift(未来函数) detected in fallback expr"
                    else:
                        expression = fallback
                        skip_reason = None
            runtime_expression = None
            runtime_error = None
            for candidate_expression in dict.fromkeys([
                expression, variant.get("expr"),
            ]):
                if not candidate_expression:
                    continue
                try:
                    parse_factor(str(candidate_expression))
                    runtime_expression = str(candidate_expression)
                    break
                except Exception as exc:
                    runtime_error = str(exc)
            if runtime_expression:
                variant["expr"] = runtime_expression
            isolation = checker._variant_isolation(variant)
            evaluation_mode = "isolation" if isolation else (
                "expression" if runtime_expression else None
            )
            if evaluation_mode is None:
                skip_reason = f"runtime_unresolved_expression:{runtime_error or skip_reason or 'unknown'}"
            runtime_variants.append(variant)
            candidates.append({
                "factor_key": key,
                "expression": runtime_expression or expression,
                "skip_reason": skip_reason,
                "evaluation_mode": evaluation_mode,
                "evaluation_signature": isolation.get("sig") if isolation else None,
                "family": get_unified_category(
                    key,
                    source=info.get("source") or variant.get("source"),
                    native_category=info.get("category") or variant.get("category"),
                    expr=expression,
                ),
                "source": info.get("source") or variant.get("source"),
                "data_safety": info.get("data_safety") or variant.get("data_safety") or "unknown",
            })
        variants = runtime_variants
        preprocessing = preprocess_candidates(candidates, min_coverage=config["min_coverage"])
        return config, variants, preprocessing

    def _run(self, run_id, config, variants, preprocessing, reuse_cache_dir=None,
             reuse_expr_overlay_dir=None):
        try:
            factor_architecture_repo.update_run(run_id, status="running")
            task_manager.update_progress(run_id, {
                "stage": "rolling_evidence", "current": 0, "total": len(variants),
                "message": "正在生成完整滚动证据",
            })
            rolling = FactorRollingService().run(
                variants,
                dt.date.fromisoformat(config["start"]),
                dt.date.fromisoformat(config["end"]),
                step_months=config["step_months"],
                lookback_months=config["train_months"],
                pool=config["pool"],
                mode="fast",
                eval_weeks=config["eval_weeks"],
                write_pool=False,
                job_id=run_id,
                min_periods_frac=1.0,
                label_horizon=config["label_horizon"],
                max_eval_dates=config["max_eval_dates"],
                train_months=config["train_months"],
                valid_months=config["valid_months"],
                test_months=config["test_months"],
                purge_gap=config["purge_gap"],
                batch_dedup=True,
                cancel_check=lambda: task_manager.is_cancel_requested(run_id),
                include_tech=False,
                reuse_cache_dir=reuse_cache_dir,
                reuse_expr_overlay_dir=reuse_expr_overlay_dir,
                progress_cb=lambda progress: task_manager.update_progress(run_id, {
                    **progress,
                    "message": progress.get("message") or "正在生成完整滚动证据",
                }),
            )
            family_map = {
                item["factor_key"]: item.get("family") or "未分类"
                for item in preprocessing["eligible"]
            }
            evidence = self._build_evidence(run_id, rolling, family_map, config)
            factor_architecture_repo.save_evidence(evidence)
            summaries = summarize_factor_evidence(evidence)
            core_candidates, core_rejected = select_core_candidates(
                summaries,
                min_folds=config["min_folds"],
                min_positive_ratio=config["min_positive_ratio"],
                min_state_count=config["min_state_count"],
                min_state_floor=config["min_state_floor"],
                min_coverage=config["min_coverage"],
            )
            core_set = build_balanced_set(
                core_candidates,
                config["core_max_count"],
                family_fraction=config["family_fraction"],
            )
            core_keys = [item["factor_key"] for item in core_set]
            set_id = f"{run_id}_core_v1"
            set_fingerprint = _fingerprint({
                "role": "core", "factor_keys": core_keys,
                "run_id": run_id, "config": config,
            })
            factor_architecture_repo.save_set(
                set_id, run_id, "core", core_keys, config,
                {
                    "candidate_count": len(core_candidates),
                    "selected_count": len(core_keys),
                    "requires_model_ab": True,
                    "summaries": core_set,
                },
                set_fingerprint,
                status="screened",
            )
            result = {
                "preprocessing": {
                    "eligible_count": preprocessing["eligible_count"],
                    "rejected_count": preprocessing["rejected_count"],
                },
                "anchors": rolling.get("anchors", []),
                "evidence_rows": len(evidence),
                "matured_fold_count": len({row["fold_id"] for row in evidence}),
                "core_candidate_count": len(core_candidates),
                "core_rejected_count": len(core_rejected),
                "core_set_id": set_id,
                "core_factor_keys": core_keys,
                "requires_model_ab": True,
                "production_pool_modified": False,
            }
            if config.get("repair_of"):
                result["repair_of"] = config["repair_of"]
                result["cache_reused"] = True
            factor_architecture_repo.update_run(run_id, status="completed", result=result)
            task_manager.update_progress(run_id, {
                "stage": "completed", "current": len(core_keys), "total": len(core_keys),
                "message": "核心候选证据筛选完成，等待模型组消融",
            })
            return result
        except TaskCancelled:
            factor_architecture_repo.update_run(run_id, status="cancelled")
            raise
        except Exception:
            error = traceback.format_exc()
            factor_architecture_repo.update_run(run_id, status="failed", error=error)
            raise

    def run_cached_repair(self, parent_run_id, cache_dir, run_id=None,
                          expr_overlay_dir=None):
        parent = factor_architecture_repo.get_run(parent_run_id)
        if not parent:
            raise ValueError("原因子架构任务不存在")
        if parent.get("status") != "completed":
            raise ValueError("仅支持修复已完成的因子架构任务")
        cache_dir = os.path.abspath(cache_dir)
        if not os.path.isfile(os.path.join(cache_dir, "manifest.json")):
            raise ValueError(f"修复缓存不完整: {cache_dir}")
        request = dict(parent.get("config") or {})
        request.update({
            "profile": "factor_architecture_cached_repair_v2",
            "repair_of": parent_run_id,
            "write_pool": False,
        })
        config, variants, preprocessing = self._normalize(request)
        config.update({
            "profile": "factor_architecture_cached_repair_v2",
            "repair_of": parent_run_id,
            "write_pool": False,
        })
        eligible = {item["factor_key"] for item in preprocessing["eligible"]}
        selected_variants = [
            variant for variant in variants
            if (variant.get("variant_key") or variant.get("template_key")) in eligible
        ]
        if not selected_variants:
            raise ValueError("修复候选池没有可评估因子")
        run_id = run_id or f"{parent_run_id}h{uuid.uuid4().hex[:6]}"
        pool_hash = _fingerprint([
            (item["factor_key"], item["expression_fingerprint"])
            for item in preprocessing["eligible"]
        ])
        factor_architecture_repo.ensure_tables()
        factor_architecture_repo.create_run(
            run_id, config, pool_hash, config["selection_cutoff"],
        )
        factor_architecture_repo.save_candidates(
            run_id, preprocessing["eligible"], preprocessing["rejected"],
        )
        result = self._run(
            run_id, config, selected_variants, preprocessing,
            reuse_cache_dir=cache_dir,
            reuse_expr_overlay_dir=expr_overlay_dir,
        )
        return {"run_id": run_id, "result": result}

    def _build_evidence(self, run_id, rolling, family_map, config, state_map=None):
        cutoff = dt.date.fromisoformat(config["selection_cutoff"])
        all_dates = []
        for anchor_result in (rolling.get("per_anchor") or {}).values():
            for detail in (anchor_result.get("detail") or {}).values():
                all_dates.extend(str(pair[0])[:10] for pair in (detail.get("daily_ics") or []))
        state_map = self._load_state_map(all_dates) if state_map is None else state_map
        rows = []
        for anchor, anchor_result in (rolling.get("per_anchor") or {}).items():
            window = anchor_result.get("window") or {}
            matured = window.get("label_matured_at")
            if not matured or dt.date.fromisoformat(matured) > cutoff:
                continue
            for key, detail in (anchor_result.get("detail") or {}).items():
                wf = detail.get("walkforward_metrics") or {}
                if wf.get("test_ic") is None:
                    continue
                direction_sign = int(wf.get("direction_sign") or 1)
                daily_pairs = [
                    (str(pair[0])[:10], float(pair[1]) * direction_sign)
                    for pair in (detail.get("daily_ics") or [])
                    if len(pair) >= 2 and pair[1] is not None
                ]
                base = {
                    "run_id": run_id,
                    "factor_key": key,
                    "family": family_map.get(key, "未分类"),
                    "fold_id": anchor,
                    "anchor_date": anchor,
                    "train_end": window.get("train_end"),
                    "valid_start": window.get("valid_start"),
                    "valid_end": window.get("valid_end"),
                    "test_start": window.get("test_start"),
                    "test_end": window.get("test_end"),
                    "label_matured_at": matured,
                    "direction_sign": direction_sign,
                    "coverage": (
                        len(daily_pairs) / max(int(wf.get("test_count") or len(daily_pairs) or 1), 1)
                    ),
                }
                all_stats = summarize_ic_values([value for _, value in daily_pairs])
                rows.append({
                    **base,
                    "state": "all",
                    "raw_ic": float(wf["test_ic"]),
                    "oriented_ic": float(wf.get("oriented_test_ic") or 0.0),
                    "icir": all_stats["icir"],
                    "positive_ratio": all_stats["positive_ratio"],
                    "sample_count": all_stats["count"],
                    "metrics": {
                        "walkforward": wf,
                        "daily_ics": daily_pairs,
                        "l1_pass": bool(wf.get("passed")),
                        "dedup_pass": bool(detail.get("all_pass")),
                    },
                })
                for state in ("high_vol", "low_vol"):
                    values = [value for date, value in daily_pairs if state_map.get(date) == state]
                    stats = summarize_ic_values(values)
                    if not values:
                        continue
                    rows.append({
                        **base,
                        "state": state,
                        "raw_ic": stats["ic_mean"] * direction_sign if stats["ic_mean"] is not None else None,
                        "oriented_ic": stats["ic_mean"],
                        "icir": stats["icir"],
                        "positive_ratio": stats["positive_ratio"],
                        "sample_count": stats["count"],
                        "metrics": {
                            "state_source": config["state_source"],
                            "l1_pass": bool(wf.get("passed")),
                            "dedup_pass": bool(detail.get("all_pass")),
                        },
                    })
        return rows

    @staticmethod
    def _load_state_map(dates):
        unique_dates = sorted(set(dates))
        if not unique_dates or not Database.is_available():
            return {}
        rows = Database.fetchall(
            "SELECT trade_date,regime FROM market_regime_history "
            "WHERE trade_date >= %s AND trade_date <= %s",
            (unique_dates[0], unique_dates[-1]),
        )
        result = {}
        for row in rows:
            regime = str(row.get("regime") or "")
            state = "high_vol" if regime.startswith("volatile") or regime == "volatile" else "low_vol"
            result[str(row["trade_date"])[:10]] = state
        return result

    def create_evidence_repair(self, parent_run_id):
        parent = factor_architecture_repo.get_run(parent_run_id)
        if not parent:
            raise ValueError("原因子架构任务不存在")
        if parent.get("status") != "completed":
            raise ValueError("仅支持修复已完成的因子架构任务")
        rolling = factor_architecture_repo.load_rolling_details(parent_run_id)
        if not rolling.get("per_anchor"):
            raise ValueError("原任务没有可复用的滚动筛选明细")
        candidates = factor_architecture_repo.list_candidates(parent_run_id)
        eligible = [item.get("metadata") or {} for item in candidates if item.get("status") == "eligible"]
        rejected = [item.get("metadata") or {} for item in candidates if item.get("status") == "rejected"]
        config = dict(parent.get("config") or {})
        repair_id = f"{parent_run_id}b{uuid.uuid4().hex[:6]}"
        config.update({
            "profile": "factor_architecture_evidence_repair_v1",
            "repair_of": parent_run_id,
            "write_pool": False,
        })
        factor_architecture_repo.create_run(
            repair_id,
            config,
            parent.get("pool_snapshot_hash") or "",
            config.get("selection_cutoff") or parent.get("selection_cutoff"),
        )
        factor_architecture_repo.save_candidates(repair_id, eligible, rejected)
        task_manager.submit_with_id(
            repair_id,
            lambda: run_with_qlib_read_access(self._run_evidence_repair,
                repair_id, parent_run_id, config, eligible, rejected, rolling,
            ),
            f"增量回补核心证据 {parent_run_id}",
            pool="default",
        )
        return {
            "run_id": repair_id,
            "parent_run_id": parent_run_id,
            "status": "pending",
            "anchors": rolling.get("anchors") or [],
            "eligible_count": len(eligible),
            "rejected_count": len(rejected),
        }

    @staticmethod
    def _repair_factor_keys(rolling, min_folds, min_positive_ratio):
        values = {}
        for anchor_result in (rolling.get("per_anchor") or {}).values():
            for key, detail in (anchor_result.get("detail") or {}).items():
                wf = detail.get("walkforward_metrics") or {}
                value = wf.get("oriented_test_ic")
                if value is not None:
                    values.setdefault(key, []).append(float(value))
        return sorted(
            key for key, factor_values in values.items()
            if len(factor_values) >= int(min_folds)
            and sum(value > 0 for value in factor_values) / len(factor_values) >= float(min_positive_ratio)
        )

    @staticmethod
    def _repair_windows(anchors, config):
        helper = OverfittingChecker()
        result = {}
        train_months = int(config.get("train_months", 60))
        valid_months = int(config.get("valid_months", 6))
        test_months = int(config.get("test_months", 3))
        purge_gap = int(config.get("purge_gap", 21))
        label_horizon = int(config.get("label_horizon", 10))
        max_eval_dates = int(config.get("max_eval_dates", 24))
        for anchor_text in anchors:
            anchor = dt.date.fromisoformat(str(anchor_text)[:10])
            train_start = _add_months(anchor, -(train_months + valid_months))
            valid_start = _add_months(anchor, -valid_months)
            test_end = _add_months(anchor, test_months) - dt.timedelta(days=1)
            calendar_dates = helper._dates_between(train_start, test_end, step=1)
            train_dates = helper._dates_between(
                train_start, valid_start - dt.timedelta(days=1), step=5,
            )
            valid_dates = helper._dates_between(
                valid_start, anchor - dt.timedelta(days=1), step=5,
            )
            test_dates = helper._dates_between(anchor, test_end, step=5)
            train_dates = purge_dates_before_boundary(
                train_dates, valid_start, calendar_dates, purge_gap,
            )
            valid_dates = purge_dates_before_boundary(
                valid_dates, anchor, calendar_dates, purge_gap,
            )
            train_dates = evenly_sample_dates(train_dates, max_eval_dates)
            valid_dates = evenly_sample_dates(valid_dates, max_eval_dates)
            test_dates = evenly_sample_dates(test_dates, max_eval_dates)
            future_dates = helper._dates_between(test_end, test_end + dt.timedelta(days=60), step=1)
            matured_at = future_dates[label_horizon] if len(future_dates) > label_horizon else None
            result[anchor.isoformat()] = {
                "test_dates": test_dates,
                "window": {
                    "train_start": train_dates[0].isoformat() if train_dates else train_start.isoformat(),
                    "train_end": train_dates[-1].isoformat() if train_dates else None,
                    "valid_start": valid_dates[0].isoformat() if valid_dates else valid_start.isoformat(),
                    "valid_end": valid_dates[-1].isoformat() if valid_dates else None,
                    "test_start": test_dates[0].isoformat() if test_dates else anchor.isoformat(),
                    "test_end": test_dates[-1].isoformat() if test_dates else test_end.isoformat(),
                    "label_matured_at": matured_at.isoformat() if matured_at else None,
                    "label_horizon": label_horizon,
                    "purge_gap": purge_gap,
                },
            }
        return result

    def _run_evidence_repair(self, run_id, parent_run_id, config, eligible, rejected, rolling):
        panel_service = FactorPanelService()
        cache_dir = os.path.join("data", "cache", "factor_evidence_repair", run_id)
        store = arrays = None
        try:
            factor_architecture_repo.update_run(run_id, status="running")
            selected_keys = self._repair_factor_keys(
                rolling, config["min_folds"], config["min_positive_ratio"],
            )
            metadata = {item.get("factor_key"): item for item in eligible if item.get("factor_key")}
            selected_keys = [key for key in selected_keys if key in metadata]
            variants = FactorRollingService.resolve_variants(
                config.get("factor_source", "pool"), config.get("source"), selected_keys,
            )
            variants_by_key = {
                variant.get("variant_key") or variant.get("template_key"): variant
                for variant in variants
            }
            for key in selected_keys:
                if key not in variants_by_key and metadata[key].get("expression"):
                    variants_by_key[key] = {
                        "variant_key": key,
                        "template_key": key,
                        "expr": metadata[key]["expression"],
                    }
            selected_keys = [key for key in selected_keys if key in variants_by_key]
            if not selected_keys:
                raise ValueError("必要条件预筛后没有可回补因子")

            windows = self._repair_windows(rolling.get("anchors") or [], config)
            cutoff = dt.date.fromisoformat(str(config["selection_cutoff"])[:10])
            windows = {
                anchor: value for anchor, value in windows.items()
                if value["window"].get("label_matured_at")
                and dt.date.fromisoformat(value["window"]["label_matured_at"]) <= cutoff
            }
            all_dates = sorted({date for value in windows.values() for date in value["test_dates"]})
            if not all_dates:
                raise ValueError("没有在选择截止日前成熟的测试窗口")

            task_manager.update_progress(run_id, {
                "stage": "shared_panel", "current": 0, "total": len(all_dates),
                "repair_of": parent_run_id, "selected_factor_count": len(selected_keys),
                "message": "正在构建测试窗共享面板",
            })

            def panel_progress(current, total, phase):
                if task_manager.is_cancel_requested(run_id):
                    raise TaskCancelled(run_id)
                task_manager.update_progress(run_id, {
                    "stage": "shared_panel", "current": current, "total": total,
                    "phase": phase, "repair_of": parent_run_id,
                    "selected_factor_count": len(selected_keys),
                    "message": "正在构建测试窗共享面板",
                })

            manifest_path, _, _ = panel_service.build_shared_memmap(
                config.get("pool", "a500"), all_dates, cache_dir,
                include_tech=False, progress_cb=panel_progress,
            )
            store, arrays, _ = panel_service.open_shared_memmap(manifest_path, all_dates)
            from core.domain.screening._screener import get_screener
            screener = get_screener()
            selected_variants = [variants_by_key[key] for key in selected_keys]
            panel_checker = OverfittingChecker()
            panel_checker.set_expr_variants(selected_variants)
            for index, eval_date in enumerate(all_dates, 1):
                if task_manager.is_cancel_requested(run_id):
                    raise TaskCancelled(run_id)
                panel_checker._ensure_date_expr_scores(
                    eval_date.isoformat(), eval_date, config.get("pool", "a500"),
                    None, store, screener,
                )
                task_manager.update_progress(run_id, {
                    "stage": "factor_panel", "current": index, "total": len(all_dates),
                    "repair_of": parent_run_id, "selected_factor_count": len(selected_keys),
                    "message": "正在预计算回补因子面板",
                })

            state_map = self._load_state_map([date.isoformat() for date in all_dates])
            industry_map = panel_checker._get_industry_map()
            family_map = {
                key: metadata[key].get("family") or "未分类" for key in selected_keys
            }
            completed = 0
            evidence_rows = 0

            def calculate_factor(key):
                checker = OverfittingChecker()
                checker.set_expr_variants([variants_by_key[key]])
                checker._date_expr_scores = panel_checker._date_expr_scores
                checker._industry_map = industry_map
                per_anchor = {}
                for anchor, plan in windows.items():
                    if task_manager.is_cancel_requested(run_id):
                        raise TaskCancelled(run_id)
                    original = ((rolling.get("per_anchor") or {}).get(anchor) or {}).get("detail", {}).get(key)
                    test_ic = (original.get("walkforward_metrics") or {}).get("test_ic") if original else None
                    if not original or test_ic is None:
                        continue
                    _, daily_pairs = checker._calc_rank_ic(
                        variants_by_key[key], plan["test_dates"], {
                            "pool": config.get("pool", "a500"),
                            "code_prefixes": None,
                            "snapshots": store,
                            "label_horizon": int(config.get("label_horizon", 10)),
                            "max_eval_dates": int(config.get("max_eval_dates", 24)),
                        },
                        return_series=True,
                        return_dates=True,
                    )
                    detail = dict(original)
                    detail["daily_ics"] = daily_pairs
                    per_anchor[anchor] = {
                        "window": plan["window"],
                        "detail": {key: detail},
                    }
                return key, {"per_anchor": per_anchor}

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(calculate_factor, key) for key in selected_keys]
                for future in as_completed(futures):
                    if task_manager.is_cancel_requested(run_id):
                        raise TaskCancelled(run_id)
                    _, factor_rolling = future.result()
                    rows = self._build_evidence(
                        run_id, factor_rolling, family_map, config, state_map=state_map,
                    )
                    factor_architecture_repo.save_evidence(rows)
                    evidence_rows += len(rows)
                    completed += 1
                    task_manager.update_progress(run_id, {
                        "stage": "daily_evidence", "current": completed,
                        "total": len(selected_keys), "evidence_rows": evidence_rows,
                        "repair_of": parent_run_id,
                        "checkpoint": "factor_fold_evidence",
                        "message": "正在补算日度与高低波证据，已逐因子落库",
                    })

            evidence = factor_architecture_repo.list_evidence(run_id)
            summaries = summarize_factor_evidence(evidence)
            core_candidates, core_rejected = select_core_candidates(
                summaries,
                min_folds=config["min_folds"],
                min_positive_ratio=config["min_positive_ratio"],
                min_state_count=config["min_state_count"],
                min_state_floor=config["min_state_floor"],
                min_coverage=config["min_coverage"],
            )
            core_set = build_balanced_set(
                core_candidates, config["core_max_count"],
                family_fraction=config["family_fraction"],
            )
            core_keys = [item["factor_key"] for item in core_set]
            set_id = f"{run_id}_core_v1"
            factor_architecture_repo.save_set(
                set_id, run_id, "core", core_keys, config,
                {
                    "candidate_count": len(core_candidates),
                    "selected_count": len(core_keys),
                    "requires_model_ab": True,
                    "summaries": core_set,
                    "repair_of": parent_run_id,
                },
                _fingerprint({
                    "role": "core", "factor_keys": core_keys,
                    "run_id": run_id, "config": config,
                }),
                status="screened",
            )
            result = {
                "repair_of": parent_run_id,
                "repair_scope_count": len(selected_keys),
                "necessary_condition_rejected_count": len(metadata) - len(selected_keys),
                "preprocessing": {
                    "eligible_count": len(eligible), "rejected_count": len(rejected),
                },
                "anchors": sorted(windows),
                "evidence_rows": len(evidence),
                "matured_fold_count": len({str(row["fold_id"]) for row in evidence}),
                "core_candidate_count": len(core_candidates),
                "core_rejected_count": len(core_rejected),
                "core_set_id": set_id,
                "core_factor_keys": core_keys,
                "requires_model_ab": True,
                "production_pool_modified": False,
                "checkpoint": "factor_fold_evidence",
            }
            factor_architecture_repo.update_run(run_id, status="completed", result=result)
            task_manager.update_progress(run_id, {
                "stage": "completed", "current": len(selected_keys),
                "total": len(selected_keys), "evidence_rows": len(evidence),
                "message": "增量证据回补完成，等待模型组消融",
            })
            return result
        except TaskCancelled:
            factor_architecture_repo.update_run(run_id, status="cancelled")
            raise
        except Exception:
            factor_architecture_repo.update_run(run_id, status="failed", error=traceback.format_exc())
            raise
        finally:
            if hasattr(store, "clear"):
                store.clear()
            if arrays is not None:
                panel_service.close_shared_memmap(arrays)
            panel_service.cleanup_shared_memmap(cache_dir)

    def create_residual(self, run_id, request=None):
        run = factor_architecture_repo.get_run(run_id)
        if not run:
            raise ValueError("因子架构任务不存在")
        sets = factor_architecture_repo.list_sets(run_id)
        core = self._preferred_set(sets, "core")
        if not core or not core.get("factor_keys"):
            raise ValueError("核心候选集不存在或为空")
        if core.get("status") != "validated" and not (request or {}).get("allow_screened_core"):
            raise ValueError("核心因子集尚未通过模型A/B，请先执行核心模型验证")
        candidates = factor_architecture_repo.list_candidates(run_id)
        core_keys = set(core["factor_keys"])
        candidate_keys = [
            item["factor_key"] for item in candidates
            if item.get("status") == "eligible" and item["factor_key"] not in core_keys
        ]
        if not candidate_keys:
            raise ValueError("没有可用于卫星筛选的剩余因子")
        task_id = f"{run_id}r{uuid.uuid4().hex[:6]}"
        task_manager.submit_with_id(
            task_id,
            lambda: run_with_qlib_read_access(
                self._run_residual, task_id, run, core, candidates, candidate_keys, request or {},
            ),
            f"风格卫星残差筛选 {len(candidate_keys)}因子",
            pool="default",
        )
        current_result = run.get("result") or {}
        current_result["active_residual_task_id"] = task_id
        factor_architecture_repo.update_run(run_id, result=current_result)
        return {
            "task_id": task_id,
            "run_id": run_id,
            "core_set_id": core["set_id"],
            "candidate_count": len(candidate_keys),
            "status": "pending",
        }

    def create_balanced_core(self, run_id, request=None):
        request = request or {}
        run = factor_architecture_repo.get_run(run_id)
        if not run:
            raise ValueError("因子架构任务不存在")
        sets = factor_architecture_repo.list_sets(run_id)
        base_set_id = str(request.get("base_set_id") or "").strip()
        base = next((item for item in sets if item.get("set_id") == base_set_id), None) if base_set_id else None
        base = base or self._preferred_set(sets, "core")
        if not base or not base.get("factor_keys"):
            raise ValueError("核心候选集不存在或为空")
        target_count = int(request.get("target_count", 15))
        if target_count < 12 or target_count > 16:
            raise ValueError("target_count 必须在12到16之间")
        existing = next((
            item for item in sets
            if item.get("role") == "core"
            and (item.get("metrics") or {}).get("variant") == "balanced_extension"
            and (item.get("metrics") or {}).get("base_set_id") == base["set_id"]
            and len(item.get("factor_keys") or []) == target_count
        ), None)
        if existing:
            return {
                "run_id": run_id,
                "set_id": existing["set_id"],
                "base_set_id": base["set_id"],
                "factor_keys": existing["factor_keys"],
                "added_factor_keys": (existing.get("metrics") or {}).get("added_factor_keys", []),
            }
        config = run.get("config") or {}
        summaries = summarize_factor_evidence(factor_architecture_repo.list_evidence(run_id))
        candidates, _ = select_core_candidates(
            summaries,
            min_folds=int(config.get("min_folds", 3)),
            min_positive_ratio=float(config.get("min_positive_ratio", 0.6)),
            min_state_count=int(config.get("min_state_count", 2)),
            min_state_floor=float(config.get("min_state_floor", -0.005)),
            min_coverage=float(config.get("min_coverage", 0.8)),
        )
        selected = build_balanced_extension(
            base["factor_keys"], candidates, target_count=target_count,
            family_fraction=float(request.get("family_fraction", 0.25)),
        )
        keys = [item["factor_key"] for item in selected]
        if len(keys) < 12:
            raise ValueError("满足稳定性条件的低相关候选不足，无法生成均衡核心集")
        version_no = max(int(item.get("version_no") or 0) for item in sets if item.get("role") == "core") + 1
        set_id = f"{run_id}_core_v{version_no}"
        added = [key for key in keys if key not in set(base["factor_keys"])]
        selection_config = {
            **config,
            "balanced_base_set_id": base["set_id"],
            "balanced_target_count": target_count,
            "family_fraction": float(request.get("family_fraction", 0.25)),
        }
        factor_architecture_repo.save_set(
            set_id, run_id, "core", keys, selection_config,
            {
                "variant": "balanced_extension",
                "base_set_id": base["set_id"],
                "added_factor_keys": added,
                "requires_fixed_oos": True,
                "summaries": selected,
            },
            _fingerprint({
                "run_id": run_id, "role": "core", "version_no": version_no,
                "base_set_id": base["set_id"], "factor_keys": keys,
            }),
            version_no=version_no,
            status="screened",
        )
        current_result = run.get("result") or {}
        current_result["balanced_core_set_id"] = set_id
        factor_architecture_repo.update_run(run_id, result=current_result)
        return {
            "run_id": run_id,
            "set_id": set_id,
            "base_set_id": base["set_id"],
            "factor_keys": keys,
            "added_factor_keys": added,
        }

    def _run_residual(self, task_id, run, core, candidates, candidate_keys, request):
        config = {**run.get("config", {}), **request, "run_id": run["run_id"]}
        task_manager.update_progress(task_id, {
            "stage": "core_oof", "current": 0, "total": 1,
            "message": "正在训练核心模型并生成滚动OOF残差",
        })
        from core.adapters.qlib.factor_set_evaluator import FactorSetEvaluator
        state_map = self._load_state_map_range(config["start"], config["selection_cutoff"])
        completed_folds = {}
        for row in factor_architecture_repo.list_evaluations(
            run["run_id"], set_id=core["set_id"],
        ):
            metrics = row.get("metrics") or {}
            if row.get("arm") == "residual_checkpoint" and row.get("fold_id") \
                    and isinstance(metrics.get("residual_rows"), list):
                completed_folds[str(row["fold_id"])] = metrics
        result = FactorSetEvaluator().evaluate_residual_candidates(
            core["factor_keys"], candidate_keys, config, state_map=state_map,
            progress_cb=lambda current, total, anchor: task_manager.update_progress(task_id, {
                "stage": (
                    "shared_prepare" if str(anchor).startswith("shared_") else "core_oof"
                ),
                "current": current, "total": total, "anchor": anchor,
                "completed_checkpoints": len(completed_folds),
                "message": (
                    "正在一次性准备共享因子数据"
                    if str(anchor).startswith("shared_")
                    else "正在训练核心模型并计算卫星残差IC"
                ),
            }),
            fold_cb=lambda anchor, core_fold, residual_rows, timing: (
                factor_architecture_repo.save_evaluations([{
                    "run_id": run["run_id"], "set_id": core["set_id"],
                    "arm": "residual_checkpoint", "fold_id": anchor,
                    "metrics": {
                        "schema_version": 1, "core_fold": core_fold,
                        "residual_rows": residual_rows, "timing": timing,
                    },
                }])
            ),
            completed_folds=completed_folds,
            cancel_cb=lambda: task_manager.is_cancel_requested(task_id),
        )
        oriented = orient_residual_rows_walkforward(result["residual_rows"])
        family_map = {
            item["factor_key"]: (item.get("metadata") or {}).get("family") or "未分类"
            for item in candidates
        }
        summary_rows = self._aggregate_residual_rows(oriented, family_map)
        summaries = summarize_factor_evidence(summary_rows)
        min_folds = int(config.get("min_folds", 3))
        min_positive_ratio = float(config.get("min_positive_ratio", 0.6))
        min_state_specificity = float(config.get("satellite_state_specificity", 0.02))
        family_overlap_limit = config.get("satellite_family_overlap_limit", 3)
        family_overlap_limit = int(family_overlap_limit) if family_overlap_limit is not None else None
        config["satellite_state_specificity"] = min_state_specificity
        config["satellite_family_overlap_limit"] = family_overlap_limit
        core_family_counts = {}
        for key in core["factor_keys"]:
            fam = family_map.get(key, "未分类")
            core_family_counts[fam] = core_family_counts.get(fam, 0) + 1
        trend, trend_rejected = select_satellite_candidates(
            summaries, "low_vol", min_folds=min_folds,
            min_positive_ratio=min_positive_ratio,
            min_state_specificity=min_state_specificity,
            core_family_counts=core_family_counts,
            family_overlap_limit=family_overlap_limit,
        )
        rebound, rebound_rejected = select_satellite_candidates(
            summaries, "high_vol", min_folds=min_folds,
            min_positive_ratio=min_positive_ratio,
            min_state_specificity=min_state_specificity,
            core_family_counts=core_family_counts,
            family_overlap_limit=family_overlap_limit,
        )
        max_count = int(config.get("satellite_max_count", 12))
        family_fraction = float(config.get("family_fraction", 0.2))
        trend_set = build_balanced_set(trend, max_count, family_fraction)
        rebound_set = build_balanced_set(rebound, max_count, family_fraction)
        existing_sets = factor_architecture_repo.list_sets(run["run_id"])
        version_by_role = {}
        for item in existing_sets:
            if item.get("role") in ("trend", "rebound"):
                version_by_role[item["role"]] = max(
                    version_by_role.get(item["role"], 0), int(item.get("version_no") or 0),
                )
        saved_sets = []
        for role, selected, rejected in (
            ("trend", trend_set, trend_rejected),
            ("rebound", rebound_set, rebound_rejected),
        ):
            keys = [item["factor_key"] for item in selected]
            version_no = version_by_role.get(role, 0) + 1
            set_id = f"{run['run_id']}_{role}_v{version_no}"
            fingerprint = _fingerprint({
                "run_id": run["run_id"], "role": role, "version_no": version_no,
                "core_set": core["set_id"], "factor_keys": keys,
            })
            factor_architecture_repo.save_set(
                set_id, run["run_id"], role, keys, config,
                {
                    "selected_count": len(keys),
                    "rejected_count": len(rejected),
                    "core_set_id": core["set_id"],
                    "requires_model_ab": True,
                    "summaries": selected,
                },
                fingerprint,
                version_no=version_no,
                status="screened",
            )
            saved_sets.append({"set_id": set_id, "role": role, "factor_keys": keys})
        evaluation_rows = [
            {
                "run_id": run["run_id"], "set_id": core["set_id"],
                "arm": "core_oof", "fold_id": fold.get("fold_id"), "metrics": fold,
            }
            for fold in result["core_folds"]
        ]
        for row in summary_rows:
            evaluation_rows.append({
                "run_id": run["run_id"], "set_id": core["set_id"],
                "arm": f"residual:{row['factor_key']}:{row['state']}",
                "fold_id": row.get("fold_id"), "metrics": row,
            })
        factor_architecture_repo.save_evaluations(evaluation_rows)
        current_result = run.get("result") or {}
        current_result["satellite_sets"] = saved_sets
        current_result["residual_evidence_rows"] = len(oriented)
        current_result["satellite_requires_model_ab"] = True
        current_result.pop("active_residual_task_id", None)
        factor_architecture_repo.update_run(run["run_id"], result=current_result)
        final = {
            "run_id": run["run_id"],
            "core_set_id": core["set_id"],
            "sets": saved_sets,
            "residual_evidence_rows": len(oriented),
        }
        task_manager.update_progress(task_id, {
            "stage": "completed", "current": len(saved_sets), "total": len(saved_sets),
            "message": "风格卫星残差筛选完成，等待因子集模型A/B",
        })
        return final

    def create_model_ab(self, run_id, request=None):
        request = request or {}
        run = factor_architecture_repo.get_run(run_id)
        if not run:
            raise ValueError("因子架构任务不存在")
        role = str(request.get("role") or "core")
        if role not in ("core", "trend", "rebound"):
            raise ValueError("role 仅支持 core/trend/rebound")
        random_count = int(request.get("random_count", 10))
        if random_count < 10 or random_count > 100:
            raise ValueError("random_count 必须在10到100之间")
        request["random_count"] = random_count
        validation_scope = str(request.get("validation_scope") or "development")
        if validation_scope not in ("development", "fixed_oos"):
            raise ValueError("validation_scope 仅支持 development/fixed_oos")
        request["validation_scope"] = validation_scope
        sets = factor_architecture_repo.list_sets(run_id)
        set_id = str(request.get("set_id") or "").strip()
        target = next((item for item in sets if item.get("set_id") == set_id), None) if set_id else None
        if set_id and not target:
            raise ValueError("指定的因子集版本不存在")
        if target and target.get("role") != role:
            raise ValueError("set_id 与 role 不匹配")
        target = target or self._preferred_set(sets, role)
        if not target or not target.get("factor_keys"):
            raise ValueError(f"{role} 因子集不存在或为空")
        core = self._preferred_set(sets, "core")
        if role != "core":
            validated_cores = [
                item for item in sets
                if item.get("role") == "core"
                and self._has_persisted_validation_evidence(run_id, item, "core", "fixed_oos")
            ]
            core = max(validated_cores, key=lambda item: int(item.get("version_no") or 0), default=None)
            if not core:
                raise ValueError("卫星模型A/B要求核心集具备当前版本固定OOS证据；旧 validated 状态不可作为验证")
        sets_to_validate = [target]
        if role != "core":
            sets_to_validate.append(core)
        elif validation_scope == "fixed_oos":
            sets_to_validate.extend(item for item in sets if item.get("role") == "core")
        self._validate_model_ab_factor_keys(run_id, sets_to_validate)
        task_id = f"{run_id}a{uuid.uuid4().hex[:6]}"
        current_result = dict(run.get("result") or {})
        current_result["active_model_ab_task_id"] = task_id
        factor_architecture_repo.update_run(run_id, result=current_result)
        try:
            task_manager.submit_with_id(
                task_id,
                lambda: run_with_qlib_read_access(self._run_model_ab, task_id, run, target, core, request),
                f"{role} 因子集模型A/B",
                pool="default",
            )
        except Exception:
            current_result.pop("active_model_ab_task_id", None)
            factor_architecture_repo.update_run(run_id, result=current_result)
            raise
        return {"task_id": task_id, "run_id": run_id, "set_id": target["set_id"], "role": role}

    def cancel(self, run_id):
        run = factor_architecture_repo.get_run(run_id)
        if not run:
            raise ValueError("因子架构任务不存在")
        result = run.get("result") or {}
        task_ids = [
            result.get("active_model_ab_task_id"),
            result.get("active_residual_task_id"),
            run_id,
        ]
        active_task_id = next(
            (item for item in task_ids if item and task_manager.is_active(item)),
            next((item for item in task_ids if item), run_id),
        )
        active = task_manager.is_active(active_task_id)
        requested = task_manager.request_cancel(active_task_id)
        if requested:
            if active:
                factor_architecture_repo.update_run(run_id, status="cancel_requested")
            else:
                task_manager.mark_cancelled(run_id)
                factor_architecture_repo.update_run(run_id, status="cancelled")
        return {"run_id": run_id, "cancel_requested": requested}

    def _run_model_ab(self, task_id, run, target, core, request):
        from core.adapters.qlib.factor_set_evaluator import FactorSetEvaluator
        from core.domain.factors.set_evaluation import analyze_core_ab, analyze_core_oos, analyze_satellite_ab

        config = {**run.get("config", {}), **request, "run_id": run["run_id"]}
        role = target["role"]
        validation_scope = config.get("validation_scope", "development")
        core_arm_sets = {}
        state_map = None
        fusion_arms = None
        if role == "core":
            candidates = factor_architecture_repo.list_candidates(run["run_id"])
            family_map = {
                item["factor_key"]: (item.get("metadata") or {}).get("family") or "未分类"
                for item in candidates
            }
            arms, family_arms = self._build_core_ab_arms(
                target["factor_keys"], candidates, family_map,
                int(config.get("random_count", 10)), int(config.get("random_seed", 20260820)),
            )
            if validation_scope == "fixed_oos":
                config = self._fixed_oos_config(run, config)
                core_sets = [
                    item for item in factor_architecture_repo.list_sets(run["run_id"])
                    if item.get("role") == "core"
                ]
                reference = min(
                    core_sets,
                    key=lambda item: int(item.get("version_no") or 0),
                )
                minimal = next((
                    item for item in core_sets
                    if int(item.get("version_no") or 0) == 2 and item.get("set_id") != target.get("set_id")
                ), None)
                arms = {
                    "core_full": arms["core_full"],
                    **({"core_reference": reference["factor_keys"]} if reference["set_id"] != target["set_id"] else {}),
                    **({"core_minimal": minimal["factor_keys"]} if minimal else {}),
                    **{key: value for key, value in arms.items() if key.startswith("random_")},
                }
                core_arm_sets = {
                    "core_full": target,
                    **({"core_reference": reference} if reference["set_id"] != target["set_id"] else {}),
                    **({"core_minimal": minimal} if minimal else {}),
                }
                family_arms = {}
        else:
            arms = {
                "core_control": core["factor_keys"],
                f"core_plus_{role}": list(dict.fromkeys(core["factor_keys"] + target["factor_keys"])),
                f"satellite_only_{role}": list(target["factor_keys"]),
            }
            fusion_arms = {
                f"core_fused_{role}": {
                    "base": "core_control",
                    "overlay": f"satellite_only_{role}",
                    "weight": float(config.get("satellite_fusion_weight", 0.5)),
                },
            }
            state_map = self._load_state_map_range(config["start"], config["end"])
            family_arms = {}
        existing_rows = factor_architecture_repo.list_evaluations(
            run["run_id"], set_id=target["set_id"],
        )
        completed_folds = {}
        for row in existing_rows:
            fold_id = row.get("fold_id")
            metrics = row.get("metrics") or {}
            if fold_id and isinstance(metrics.get("daily_ics"), list):
                completed_folds.setdefault(row["arm"], {})[str(fold_id)] = metrics
        completed_count = sum(
            len(folds) for arm, folds in completed_folds.items() if arm in arms
        )
        task_manager.update_progress(task_id, {
            "stage": "model_ab", "current": 0, "total": len(arms),
            "completed_checkpoints": completed_count,
            "message": f"正在执行{len(arms)}组固定参数滚动模型实验，已复用{completed_count}折",
        })
        results = FactorSetEvaluator().evaluate_factor_sets(
            arms, config,
            progress_cb=lambda current, total, arm, anchor: task_manager.update_progress(task_id, {
                "stage": "model_ab", "current": current, "total": total,
                "arm": arm, "anchor": anchor, "message": "正在执行滚动模型A/B",
            }),
            fold_cb=lambda arm, fold: factor_architecture_repo.save_evaluations([{
                "run_id": run["run_id"], "set_id": target["set_id"], "arm": arm,
                "fold_id": fold.get("fold_id"), "metrics": fold,
            }]),
            completed_folds=completed_folds,
            cancel_cb=lambda: task_manager.is_cancel_requested(task_id),
            state_map=state_map,
            fusion_arms=fusion_arms,
        )
        validation_policy = _validation_policy(role, validation_scope, config)
        if role == "core" and validation_scope == "fixed_oos":
            analysis = analyze_core_oos(
                results,
                min_folds=validation_policy["min_folds"],
                min_positive_ratio=validation_policy["min_positive_ratio"],
                min_random_percentile=validation_policy["min_random_percentile"],
                min_random_count=validation_policy["min_random_count"],
            )
            analysis["period"] = {"start": config["start"], "end": config["end"]}
            winner_set = core_arm_sets.get(analysis.get("winner_arm"))
            analysis["winner_set_id"] = winner_set.get("set_id") if winner_set else None
        elif role == "core":
            analysis = analyze_core_ab(
                results, family_arms,
                min_folds=validation_policy["min_folds"],
                min_positive_ratio=validation_policy["min_positive_ratio"],
                min_random_percentile=validation_policy["min_random_percentile"],
                min_random_count=validation_policy["min_random_count"],
            )
        else:
            analysis = analyze_satellite_ab(
                results, role, min_folds=validation_policy["min_folds"],
                min_positive_ratio=validation_policy["min_positive_ratio"],
                min_median_delta=validation_policy["min_median_delta"],
                min_unconditional_delta=validation_policy["min_unconditional_delta"],
            )
            satellite_only = results.get(f"satellite_only_{role}") or {}
            fused = results.get(f"core_fused_{role}") or {}
            analysis["satellite_only"] = {
                "rank_ic": satellite_only.get("rank_ic"),
                "icir": satellite_only.get("icir"),
                "positive_ratio": satellite_only.get("positive_ratio"),
                "n_dates": satellite_only.get("n_dates"),
            }
            analysis["fusion"] = {
                "rank_ic": fused.get("rank_ic"),
                "icir": fused.get("icir"),
                "positive_ratio": fused.get("positive_ratio"),
                "n_dates": fused.get("n_dates"),
                "weight": float(config.get("satellite_fusion_weight", 0.5)),
            }
        refined_set = None
        if role == "core" and validation_scope == "development" and analysis.get("harmful_families"):
            harmful = set(analysis["harmful_families"])
            refined_keys = [
                key for key in target["factor_keys"]
                if family_map.get(key, "未分类") not in harmful
            ]
            if refined_keys and refined_keys != target["factor_keys"]:
                version_no = int(target.get("version_no") or 1) + 1
                refined_set_id = f"{run['run_id']}_core_v{version_no}"
                factor_architecture_repo.save_set(
                    refined_set_id, run["run_id"], "core", refined_keys, config,
                    {
                        "refined_from": target["set_id"],
                        "removed_families": sorted(harmful),
                        "requires_model_ab": True,
                    },
                    _fingerprint({
                        "run_id": run["run_id"], "role": "core",
                        "version_no": version_no, "factor_keys": refined_keys,
                    }),
                    version_no=version_no,
                    status="screened",
                )
                refined_set = {"set_id": refined_set_id, "factor_keys": refined_keys}
                analysis["refined_set"] = refined_set
                try:
                    balanced = self.create_balanced_core(
                        run["run_id"], {"base_set_id": refined_set_id, "target_count": 15},
                    )
                    refined_set = {
                        "set_id": balanced["set_id"],
                        "factor_keys": balanced["factor_keys"],
                    }
                    analysis["balanced_set"] = refined_set
                except ValueError as exc:
                    analysis["balanced_error"] = str(exc)
        metrics = {**(target.get("metrics") or {}), "model_ab": analysis}
        metrics.pop("validation_evidence", None)
        if analysis.get("validated") is True:
            metrics["validation_evidence"] = _build_validation_evidence(
                run["run_id"], target, role, analysis.get("scope") or validation_scope,
                analysis, config, results,
            )
        final_set = None
        if role == "core" and validation_scope == "fixed_oos":
            winner_set = core_arm_sets.get(analysis.get("winner_arm"))
            for arm, set_item in core_arm_sets.items():
                item_metrics = {
                    **(set_item.get("metrics") or {}),
                    "final_oos": {
                        "selected": bool(winner_set and set_item["set_id"] == winner_set["set_id"]),
                        "arm": arm,
                        "analysis": analysis,
                    },
                }
                item_status = (
                    "validated"
                    if winner_set and set_item["set_id"] == winner_set["set_id"]
                    else "superseded"
                )
                item_metrics.pop("validation_evidence", None)
                if item_status == "validated":
                    item_metrics["validation_evidence"] = _build_validation_evidence(
                        run["run_id"], set_item, "core", "fixed_oos", analysis, config, results,
                        evaluation_set_id=target["set_id"],
                    )
                factor_architecture_repo.update_set_status(set_item["set_id"], item_status, item_metrics)
            status = "validated" if winner_set else "rejected"
            if not winner_set:
                factor_architecture_repo.update_set_status(target["set_id"], "rejected", metrics)
            final_set = winner_set
        elif role == "core":
            status = "screened"
            target_status = "superseded" if refined_set else "screened"
            factor_architecture_repo.update_set_status(target["set_id"], target_status, metrics)
        else:
            status = "validated" if analysis["validated"] else "rejected"
            factor_architecture_repo.update_set_status(target["set_id"], status, metrics)
        rows = []
        for arm, arm_result in results.items():
            rows.append({
                "run_id": run["run_id"], "set_id": target["set_id"], "arm": arm,
                "fold_id": None, "metrics": {key: value for key, value in arm_result.items() if key != "folds"},
            })
            rows.extend({
                "run_id": run["run_id"], "set_id": target["set_id"], "arm": arm,
                "fold_id": fold.get("fold_id"), "metrics": fold,
            } for fold in arm_result.get("folds", []))
        factor_architecture_repo.save_evaluations(rows)
        latest_run = factor_architecture_repo.get_run(run["run_id"]) or run
        current_result = latest_run.get("result") or {}
        current_result.setdefault("model_ab", {})[role] = {
            "set_id": (final_set or target)["set_id"], "status": status, "analysis": analysis,
        }
        if current_result.get("active_model_ab_task_id") == task_id:
            current_result.pop("active_model_ab_task_id", None)
        if role == "core" and validation_scope == "fixed_oos":
            current_result["final_core_set_id"] = final_set["set_id"] if final_set else None
        factor_architecture_repo.update_run(run["run_id"], result=current_result)
        if role == "core" and validation_scope == "development":
            completion_message = "核心组合优化完成，内部候选已锁定，等待最终OOS验证"
        elif role == "core" and final_set:
            completion_message = f"最终核心集已确定：{len(final_set['factor_keys'])}因子"
        elif role == "core":
            completion_message = "最终OOS未通过，没有可投入后续流程的核心集"
        else:
            completion_message = f"{role}模型A/B完成：{status}"
        task_manager.update_progress(task_id, {
            "stage": "completed", "current": len(arms), "total": len(arms),
            "message": completion_message,
        })
        return {
            "run_id": run["run_id"], "set_id": target["set_id"], "status": status,
            "analysis": analysis, "refined_set": refined_set,
            "final_set_id": final_set["set_id"] if final_set else None,
        }

    @staticmethod
    def _fixed_oos_config(run, config):
        from core.adapters.qlib.factor_set_evaluator import _make_rounds, _matured

        development_rounds = _make_rounds(run.get("config") or {})
        if not development_rounds:
            raise ValueError("原始开发期没有可用滚动窗口")
        step_months = int(config.get("step_months", 3))
        next_anchor = _add_months(dt.date.fromisoformat(development_rounds[-1][0]), step_months)
        cutoff = dt.date.today()
        probe = {
            **config,
            "start": next_anchor.isoformat(),
            "end": cutoff.isoformat(),
            "selection_cutoff": cutoff.isoformat(),
        }
        matured = [
            item for item in _make_rounds(probe)
            if _matured(item[3][1], int(probe.get("label_horizon", 10))) <= cutoff
        ]
        if len(matured) < int(config.get("min_folds", 3)):
            raise ValueError("固定样本外窗口尚未成熟到足够折数")
        return {
            **probe,
            "start": matured[0][0],
            "end": matured[-1][0],
            "fixed_oos": True,
        }

    @staticmethod
    def _build_core_ab_arms(core_keys, candidates, family_map, random_count, seed):
        from core.domain.factors.expression import parse_factor

        arms = {"core_full": list(core_keys)}
        family_arms = {}
        for family in sorted({family_map.get(key, "未分类") for key in core_keys}):
            remaining = [key for key in core_keys if family_map.get(key, "未分类") != family]
            if not remaining:
                continue
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", family).strip("_") or hashlib.md5(
                family.encode("utf-8")
            ).hexdigest()[:8]
            arm = f"minus_{slug}"[:32]
            arms[arm] = remaining
            family_arms[family] = arm
        eligible_by_family = {}
        for item in candidates:
            if item.get("status") != "eligible":
                continue
            key = item["factor_key"]
            metadata = item.get("metadata") or {}
            expression = metadata.get("expression")
            if not expression:
                expression, _ = resolve_factor_expression(key)
            if not expression:
                continue
            try:
                parse_factor(str(expression))
            except Exception:
                continue
            eligible_by_family.setdefault(family_map.get(key, "未分类"), []).append(key)
        rng = random.Random(seed)
        core_family_counts = {}
        for key in core_keys:
            family = family_map.get(key, "未分类")
            core_family_counts[family] = core_family_counts.get(family, 0) + 1
        for index in range(max(0, random_count)):
            sample = []
            for family, count in core_family_counts.items():
                pool = eligible_by_family.get(family, [])
                if len(pool) >= count:
                    sample.extend(rng.sample(pool, count))
            sample = list(dict.fromkeys(sample))
            if len(sample) == len(core_keys):
                arms[f"random_{index + 1:02d}"] = sample
        return arms, family_arms

    @staticmethod
    def _preferred_set(sets, role):
        matches = [item for item in sets if item.get("role") == role]
        priority = {"validated": 3, "screened": 2, "draft": 1, "rejected": 0}
        return max(
            matches,
            key=lambda item: (priority.get(item.get("status"), -1), int(item.get("version_no") or 0)),
            default=None,
        )

    @staticmethod
    def _has_persisted_validation_evidence(run_id, factor_set, role, scope):
        if not factor_set or not _has_current_validation_evidence(factor_set, run_id, role, scope):
            return False
        evidence = (factor_set.get("metrics") or {}).get("validation_evidence") or {}
        evaluation_set_id = evidence.get("evaluation_set_id")
        expected = evidence.get("folds_by_arm") or {}
        expected_fingerprints = evidence.get("arm_fingerprints") or {}
        if not evaluation_set_id or not expected or set(expected) != set(expected_fingerprints):
            return False
        try:
            rows = factor_architecture_repo.list_evaluations(run_id, set_id=evaluation_set_id)
        except Exception:
            return False
        summaries = {}
        folds = {}
        for row in rows:
            arm = row.get("arm")
            if not arm:
                return False
            if row.get("fold_id") is None:
                if arm in summaries:
                    return False
                summaries[arm] = row.get("metrics")
            else:
                folds.setdefault(arm, []).append(row.get("metrics") or {})
        if set(summaries) != set(expected) or set(folds) - set(expected):
            return False
        results = {}
        for arm, fold_ids in expected.items():
            summary = summaries.get(arm)
            arm_folds = sorted(folds.get(arm, []), key=lambda item: str(item.get("fold_id") or ""))
            actual_ids = [str(item.get("fold_id")) for item in arm_folds if item.get("fold_id") is not None]
            if not isinstance(summary, dict) or actual_ids != list(fold_ids):
                return False
            fingerprint = _fingerprint({
                "factor_keys": summary.get("factor_keys") or [],
                "summary": summary,
                "folds": arm_folds,
            })
            if fingerprint != expected_fingerprints.get(arm):
                return False
            results[arm] = {**summary, "folds": arm_folds}

        policy = evidence["policy"]
        analysis = evidence["analysis"]
        from core.domain.factors.set_evaluation import analyze_core_ab, analyze_core_oos, analyze_satellite_ab

        if role == "core" and scope == "fixed_oos":
            actual_analysis = analyze_core_oos(
                results,
                min_folds=policy["min_folds"],
                min_positive_ratio=policy["min_positive_ratio"],
                min_random_percentile=policy["min_random_percentile"],
                min_random_count=policy["min_random_count"],
            )
            return (
                actual_analysis.get("validated") is True
                and actual_analysis.get("winner_arm") == analysis.get("winner_arm")
                and actual_analysis.get("candidate_results") == analysis.get("candidate_results")
                and actual_analysis.get("random_count") == analysis.get("random_count")
            )
        if role == "core" and scope == "development":
            family_arms = {
                family: item.get("arm")
                for family, item in (analysis.get("family_results") or {}).items()
            }
            actual_analysis = analyze_core_ab(
                results, family_arms,
                min_folds=policy["min_folds"],
                min_positive_ratio=policy["min_positive_ratio"],
                min_random_percentile=policy["min_random_percentile"],
                min_random_count=policy["min_random_count"],
            )
            return actual_analysis.get("validated") is True and actual_analysis == analysis
        if role in ("trend", "rebound") and scope == "state_conditional":
            actual_analysis = analyze_satellite_ab(
                results, role,
                min_folds=policy["min_folds"],
                min_positive_ratio=policy["min_positive_ratio"],
                min_median_delta=policy["min_median_delta"],
                min_unconditional_delta=policy["min_unconditional_delta"],
            )
            return (
                actual_analysis.get("validated") is True
                and all(actual_analysis.get(key) == analysis.get(key) for key in (
                    "role", "scope", "target_state", "fold_count", "median_delta", "mean_delta",
                    "positive_ratio", "fold_deltas", "state_days", "unconditional",
                ))
            )
        return False

    @staticmethod
    def _validate_model_ab_factor_keys(run_id, sets):
        candidates = {
            item.get("factor_key"): item
            for item in factor_architecture_repo.list_candidates(run_id)
        }
        invalid = []
        seen_invalid = set()
        from core.domain.factors.expression import parse_factor

        for factor_set in sets:
            if not factor_set:
                continue
            for key in factor_set.get("factor_keys") or []:
                candidate = candidates.get(key) or {}
                metadata = candidate.get("metadata") or {}
                if not candidate and not metadata:
                    item = (key, "当前运行未找到候选元数据")
                    if item not in seen_invalid:
                        invalid.append(item)
                        seen_invalid.add(item)
                    continue
                data_safety = str(metadata.get("data_safety") or "unknown").lower()
                if data_safety not in ("safe", "pit", "point_in_time"):
                    item = (key, f"data_safety={data_safety}")
                    if item not in seen_invalid:
                        invalid.append(item)
                        seen_invalid.add(item)
                    continue
                expression, reason = resolve_factor_expression(key)
                if not expression:
                    expression = metadata.get("expression")
                if not expression:
                    item = (key, reason or candidate.get("reject_reason") or "无可用表达式")
                    if item not in seen_invalid:
                        invalid.append(item)
                        seen_invalid.add(item)
                    continue
                try:
                    parse_factor(str(expression))
                except Exception as exc:
                    item = (key, str(exc))
                    if item not in seen_invalid:
                        invalid.append(item)
                        seen_invalid.add(item)

        if invalid:
            details = "、".join(f"{key}({reason})" for key, reason in invalid[:8])
            raise ValueError(
                f"因子集包含无法用于模型A/B的因子: {details}。"
                "该因子集可能由旧版预筛产生，请重新运行因子架构筛选"
            )

    @staticmethod
    def _aggregate_residual_rows(rows, family_map):
        groups = {}
        for row in rows:
            key = (row["factor_key"], str(row["fold_id"]), row.get("state") or "unknown")
            groups.setdefault(key, []).append(float(row["oriented_ic"]))
            all_key = (row["factor_key"], str(row["fold_id"]), "all")
            groups.setdefault(all_key, []).append(float(row["oriented_ic"]))
        result = []
        for (factor_key, fold_id, state), values in groups.items():
            stats = summarize_ic_values(values)
            result.append({
                "factor_key": factor_key,
                "family": family_map.get(factor_key, "未分类"),
                "fold_id": fold_id,
                "state": state,
                "oriented_ic": stats["ic_mean"],
                "coverage": 1.0,
                "icir": stats["icir"],
                "positive_ratio": stats["positive_ratio"],
                "sample_count": stats["count"],
            })
        return result

    @staticmethod
    def _load_state_map_range(start, end):
        if not Database.is_available():
            return {}
        rows = Database.fetchall(
            "SELECT trade_date,regime FROM market_regime_history "
            "WHERE trade_date >= %s AND trade_date <= %s", (start, end),
        )
        result = {}
        for row in rows:
            regime = str(row.get("regime") or "")
            result[str(row["trade_date"])[:10]] = (
                "high_vol" if regime.startswith("volatile") or regime == "volatile" else "low_vol"
            )
        return result

    def publish_satellites(self, run_id):
        from core.domain.factors.registry import get_pool

        run = factor_architecture_repo.get_run(run_id)
        if not run:
            raise ValueError("因子架构任务不存在")
        sets = factor_architecture_repo.list_sets(run_id)

        def best(role, scope):
            candidates = [
                item for item in sets
                if item.get("role") == role
                and item.get("status") == "validated"
                and self._has_persisted_validation_evidence(run_id, item, role, scope)
            ]
            return max(
                candidates,
                key=lambda item: int(item.get("version_no") or 0),
                default=None,
            )

        core = best("core", "fixed_oos")
        trend = best("trend", "state_conditional")
        rebound = best("rebound", "state_conditional")
        if not core:
            raise ValueError("核心因子集缺少当前版本固定OOS证据；旧 validated 状态不可发布")
        if not trend and not rebound:
            raise ValueError("没有具备当前版本模型A/B证据的卫星集可发布")
        candidates = factor_architecture_repo.list_candidates(run_id)
        self._validate_model_ab_factor_keys(run_id, [core, trend, rebound])
        candidate_map = {item.get("factor_key"): item for item in candidates}
        family_map = {
            item["factor_key"]: (item.get("metadata") or {}).get("family") or "未分类"
            for item in candidates
        }
        specs = [
            ("core_base", "核心基底", core["factor_keys"], core["set_id"], [core]),
            ("trend_satellite", "趋势卫星", (trend or {}).get("factor_keys") or [], (trend or {}).get("set_id"), [trend]),
            ("rebound_satellite", "反弹卫星", (rebound or {}).get("factor_keys") or [], (rebound or {}).get("set_id"), [rebound]),
        ]
        _core_keys = core["factor_keys"]
        _trend_keys = (trend or {}).get("factor_keys") or []
        _rebound_keys = (rebound or {}).get("factor_keys") or []
        if _trend_keys:
            specs.append((
                "core_trend_satellite", "核心+趋势卫星(融合)",
                list(dict.fromkeys(_core_keys + _trend_keys)),
                f"{core['set_id']}+{(trend or {}).get('set_id', '')}",
                [core, trend],
            ))
        if _rebound_keys:
            specs.append((
                "core_rebound_satellite", "核心+反弹卫星(融合)",
                list(dict.fromkeys(_core_keys + _rebound_keys)),
                f"{core['set_id']}+{(rebound or {}).get('set_id', '')}",
                [core, rebound],
            ))
        pool = get_pool()
        published = []
        for name, label, keys, set_id, source_sets in specs:
            if not keys:
                continue
            by_category = {}
            for key in keys:
                category = family_map.get(key, "未分类")
                by_category.setdefault(category, []).append(key)
            factors_by_category = {
                category: [(key, key, None, None) for key in category_keys]
                for category, category_keys in by_category.items()
            }
            factor_specs = {}
            for key in keys:
                metadata = (candidate_map.get(key) or {}).get("metadata") or {}
                factor_specs[key] = {
                    "label": metadata.get("label") or key,
                    "template_key": metadata.get("template_key"),
                    "params": metadata.get("params") or {},
                    "expr": metadata.get("expression") or metadata.get("expr"),
                    "data_safety": metadata.get("data_safety") or "unknown",
                    "source": metadata.get("source"),
                    "active": True,
                }
            attestations = [
                (factor_set.get("metrics") or {}).get("validation_evidence")
                for factor_set in source_sets if factor_set
            ]
            pool.save_curated_set(
                name,
                factors_by_category,
                config={
                    "description": f"{label}（核心/卫星架构成品 · {run_id}）",
                    "source_run_id": run_id,
                    "source_set_id": set_id,
                    "qualification": "research_only",
                    "production_eligible": False,
                },
                stats={"total": len(keys), "source_run_id": run_id},
                factor_specs=factor_specs,
                qualification="research_only",
                validation_evidence={"source_sets": attestations},
            )
            published.append({
                "name": name, "label": label, "count": len(keys), "set_id": set_id,
                "qualification": "research_only", "production_eligible": False,
            })
        return {"run_id": run_id, "published": published}

    @staticmethod
    def get(run_id):
        run = factor_architecture_repo.get_run(run_id)
        if not run:
            return None
        task = task_manager.get_dict(run_id) or {}
        result = run.get("result") or {}
        active_task_id = result.get("active_model_ab_task_id") or result.get("active_residual_task_id")
        active_subtask = task_manager.get_dict(active_task_id) if active_task_id else None
        return {
            **run,
            "status": task.get("status") or run.get("status"),
            "progress": task.get("progress") or {},
            "active_subtask": active_subtask,
            "sets": factor_architecture_repo.list_sets(run_id),
        }


factor_architecture_service = FactorArchitectureService()
