import hashlib
import json
import traceback
import uuid

from core.data.repos.factor_architecture import factor_architecture_repo
from core.data.repos.regime_history import regime_history_repo
from core.domain.factors.evidence import regroup_evidence_by_state, summarize_factor_evidence
from core.domain.factors.role_selection import build_balanced_set, select_core_candidates
from core.shared.infrastructure.async_task import TaskCancelled, task_manager


def _fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FactorStateRegroupService:
    def create(self, parent_run_id):
        parent = factor_architecture_repo.get_run(parent_run_id)
        if not parent or parent.get("status") != "completed":
            raise ValueError("仅支持重新分组已完成的因子架构任务")
        all_rows = [
            row for row in factor_architecture_repo.list_evidence(parent_run_id)
            if row.get("state") == "all"
        ]
        if not all_rows:
            raise ValueError("原任务没有可重分组的日度IC证据")
        candidates = factor_architecture_repo.list_candidates(parent_run_id)
        eligible = [item.get("metadata") or {} for item in candidates if item.get("status") == "eligible"]
        rejected = [item.get("metadata") or {} for item in candidates if item.get("status") == "rejected"]
        run_id = f"{parent_run_id}s{uuid.uuid4().hex[:5]}"
        config = {
            **(parent.get("config") or {}),
            "profile": "factor_architecture_state_regroup_v1",
            "state_regroup_of": parent_run_id,
            "state_source": "market_regime_history_volatility_axis_daily_v1",
            "write_pool": False,
        }
        factor_architecture_repo.ensure_tables()
        factor_architecture_repo.create_run(
            run_id, config, parent.get("pool_snapshot_hash") or "",
            config.get("selection_cutoff") or parent.get("selection_cutoff"),
        )
        factor_architecture_repo.save_candidates(run_id, eligible, rejected)
        task_manager.submit_with_id(
            run_id,
            lambda: self._run(run_id, parent_run_id, config, eligible, all_rows),
            f"重新分组高低波证据 {parent_run_id}",
            pool="default",
        )
        return {"run_id": run_id, "parent_run_id": parent_run_id, "status": "pending"}

    def _run(self, run_id, parent_run_id, config, eligible, all_rows):
        try:
            factor_architecture_repo.update_run(run_id, status="running")
            dates = sorted({
                str(pair[0])[:10]
                for row in all_rows
                for pair in ((row.get("metrics") or {}).get("daily_ics") or [])
                if len(pair) >= 2
            })
            if not dates:
                raise RuntimeError("日度IC日期为空")
            regime_history_repo.ensure_schema()
            state_map = regime_history_repo.get_state_map(dates[0], dates[-1])
            mapped_dates = {date for date in dates if date in state_map}
            coverage = len(mapped_dates) / len(dates)
            high_dates = sum(state_map.get(date) == "high_vol" for date in dates)
            low_dates = sum(state_map.get(date) == "low_vol" for date in dates)
            if coverage < 0.95:
                raise RuntimeError(f"历史状态日期覆盖率不足: {coverage:.1%} < 95%")
            if not high_dates or not low_dates:
                raise RuntimeError("历史状态未同时覆盖high_vol和low_vol")
            family_map = {
                item.get("factor_key"): item.get("family") or "未分类"
                for item in eligible if item.get("factor_key")
            }
            grouped = {}
            for row in all_rows:
                grouped.setdefault(row["factor_key"], []).append(row)
            written = 0
            for index, (key, factor_rows) in enumerate(sorted(grouped.items()), 1):
                if task_manager.is_cancel_requested(run_id):
                    raise TaskCancelled(run_id)
                rows = regroup_evidence_by_state(
                    factor_rows, state_map, run_id, family_map=family_map,
                )
                factor_architecture_repo.save_evidence(rows)
                written += len(rows)
                task_manager.update_progress(run_id, {
                    "stage": "regroup", "current": index, "total": len(grouped),
                    "evidence_rows": written, "state_date_coverage": coverage,
                    "high_vol_dates": high_dates, "low_vol_dates": low_dates,
                    "checkpoint": "factor_fold_evidence",
                    "message": "正在使用逐日波动状态重新分组现有IC证据",
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
                    "state_regroup_of": parent_run_id,
                },
                _fingerprint({"role": "core", "factor_keys": core_keys, "run_id": run_id, "config": config}),
                status="screened",
            )
            result = {
                "state_regroup_of": parent_run_id,
                "state_date_count": len(dates),
                "state_date_coverage": coverage,
                "high_vol_dates": high_dates,
                "low_vol_dates": low_dates,
                "evidence_rows": len(evidence),
                "matured_fold_count": len({str(row["fold_id"]) for row in evidence}),
                "core_candidate_count": len(core_candidates),
                "core_rejected_count": len(core_rejected),
                "core_set_id": set_id,
                "core_factor_keys": core_keys,
                "requires_model_ab": True,
                "production_pool_modified": False,
            }
            factor_architecture_repo.update_run(run_id, status="completed", result=result)
            task_manager.update_progress(run_id, {
                "stage": "completed", "current": len(grouped), "total": len(grouped),
                "evidence_rows": len(evidence),
                "message": "高低波证据重分组与核心候选重算完成",
            })
            return result
        except TaskCancelled:
            factor_architecture_repo.update_run(run_id, status="cancelled")
            raise
        except Exception:
            factor_architecture_repo.update_run(run_id, status="failed", error=traceback.format_exc())
            raise


factor_state_regroup_service = FactorStateRegroupService()
