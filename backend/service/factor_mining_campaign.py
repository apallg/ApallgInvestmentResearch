import datetime as dt
import threading
import time
import traceback
import uuid

import psutil

from backend.service.factor_mining import factor_mining_service
from core.data.repos.factor_mining import factor_mining_repo
from core.domain.factors.archive import campaign_seed, rank_hall_of_fame
from core.shared.infrastructure.async_task import TaskCancelled, task_manager


class FactorMiningCampaignService:
    def create(self, config):
        normalized = self._normalize(config)
        campaign_id = uuid.uuid4().hex[:12]
        progress = self._progress("waiting_resources", 0, normalized)
        factor_mining_repo.ensure_tables()
        factor_mining_service._ensure_archive()
        factor_mining_repo.create_campaign(campaign_id, normalized, progress)
        task_manager.submit_with_id(
            campaign_id,
            lambda: self._run(campaign_id, normalized),
            f"持续因子挖掘 {normalized['pool']}",
            pool="light",
        )
        return {"campaign_id": campaign_id, "status": "pending", "progress": progress}

    @staticmethod
    def _normalize(config):
        base = factor_mining_service._normalize_config({**config, "defer_final_oos": True})
        base.update({
            "max_batches": int(config.get("max_batches", 0)),
            "max_hours": float(config.get("max_hours", 0)),
            "idle_cpu_percent": float(config.get("idle_cpu_percent", 70)),
            "min_free_memory_gb": float(config.get("min_free_memory_gb", 6)),
            "resource_poll_seconds": int(config.get("resource_poll_seconds", 15)),
        })
        if base["max_batches"] < 0 or base["max_hours"] < 0:
            raise ValueError("max_batches 和 max_hours 不能为负数，0 表示直到手工终止")
        if not 10 <= base["idle_cpu_percent"] <= 100:
            raise ValueError("idle_cpu_percent 必须在 10 到 100 之间")
        if base["min_free_memory_gb"] < 1:
            raise ValueError("min_free_memory_gb 不能低于 1GB")
        if not 3 <= base["resource_poll_seconds"] <= 60:
            raise ValueError("resource_poll_seconds 必须在 3 到 60 秒之间")
        return base

    @staticmethod
    def _resource_state(config):
        memory = psutil.virtual_memory()
        cpu = float(psutil.cpu_percent(interval=None))
        free_gb = float(memory.available / 1024 ** 3)
        return {
            "cpu_percent": round(cpu, 1),
            "free_memory_gb": round(free_gb, 2),
            "ready": cpu <= config["idle_cpu_percent"] and free_gb >= config["min_free_memory_gb"],
        }

    @staticmethod
    def _progress(stage, batch_index, config, **extra):
        return {
            "stage": stage,
            "current": batch_index,
            "total": config["max_batches"],
            "batch_index": batch_index,
            "unique_candidate_count": factor_mining_repo.archive_count(),
            "heartbeat_at": dt.datetime.now().isoformat(timespec="seconds"),
            **extra,
        }

    def _run(self, campaign_id, config):
        started = time.monotonic()
        batch_index = 0
        current_run_id = ""
        try:
            while True:
                task_manager.check_cancelled(campaign_id)
                elapsed_hours = (time.monotonic() - started) / 3600
                if config["max_batches"] and batch_index >= config["max_batches"]:
                    break
                if config["max_hours"] and elapsed_hours >= config["max_hours"]:
                    break

                resources = self._resource_state(config)
                if not resources["ready"]:
                    progress = self._progress(
                        "waiting_resources", batch_index, config,
                        resource_state=resources,
                        message="等待 CPU 和可用内存满足启动条件",
                    )
                    factor_mining_repo.update_campaign(campaign_id, status="waiting_resources", progress=progress)
                    task_manager.update_progress(campaign_id, progress)
                    threading.Event().wait(min(config["resource_poll_seconds"], 30))
                    continue

                batch_index += 1
                seed = campaign_seed(config["seed"], campaign_id, batch_index)
                child_config = {
                    **config,
                    "seed": seed,
                    "campaign_id": campaign_id,
                    "batch_index": batch_index,
                    "defer_final_oos": True,
                }
                child = factor_mining_service.create_run(child_config)
                current_run_id = child["run_id"]
                factor_mining_repo.save_campaign_run(
                    campaign_id, batch_index, current_run_id, seed, "running")
                while True:
                    if task_manager.is_cancel_requested(campaign_id):
                        factor_mining_service.cancel(current_run_id)
                        raise TaskCancelled(campaign_id)
                    run = factor_mining_service.get_run(current_run_id) or {}
                    status = run.get("status", "pending")
                    child_progress = run.get("progress") or {}
                    progress = self._progress(
                        "mining_batch", batch_index, config,
                        current_run_id=current_run_id,
                        child_status=status,
                        child_progress=child_progress,
                        candidate_count=child_progress.get("candidate_count", 0),
                        passed_count=child_progress.get("passed_count", 0),
                        best_train_ic=child_progress.get("best_train_ic"),
                        best_oos_ic=child_progress.get("best_oos_ic"),
                        last_date=child_progress.get("last_date"),
                        resource_state=resources,
                        message=f"正在执行第 {batch_index} 批",
                    )
                    factor_mining_repo.update_campaign(campaign_id, status="running", progress=progress)
                    task_manager.update_progress(campaign_id, progress)
                    if status in ("completed", "awaiting_promotion", "failed", "cancelled"):
                        summary = (run.get("result") or {}).get("summary") or (run.get("report") or {}).get("summary") or {}
                        factor_mining_repo.save_campaign_run(
                            campaign_id, batch_index, current_run_id, seed, status, summary)
                        break
                    threading.Event().wait(2)
                current_run_id = ""

            progress = self._progress(
                "completed", batch_index, config, message="持续挖掘已达到运行预算")
            factor_mining_repo.update_campaign(campaign_id, status="completed", progress=progress)
            return {"campaign_id": campaign_id, "summary": progress}
        except TaskCancelled:
            if current_run_id:
                factor_mining_service.cancel(current_run_id)
            progress = self._progress(
                "cancelled", batch_index, config, current_run_id=current_run_id,
                message="持续挖掘已终止")
            factor_mining_repo.update_campaign(campaign_id, status="cancelled", progress=progress)
            raise
        except Exception:
            error = traceback.format_exc()
            progress = self._progress(
                "failed", batch_index, config, current_run_id=current_run_id,
                message="持续挖掘失败")
            factor_mining_repo.update_campaign(
                campaign_id, status="failed", progress=progress, error=error)
            raise

    def get(self, campaign_id):
        persisted = factor_mining_repo.get_campaign(campaign_id)
        if not persisted:
            return None
        task = task_manager.get_dict(campaign_id) or {}
        runs = factor_mining_repo.list_campaign_runs(campaign_id)
        candidates = []
        for item in runs:
            candidates.extend(factor_mining_repo.list_candidates(item["run_id"]))
        return {
            **persisted,
            "status": task.get("status") or persisted["status"],
            "progress": task.get("progress") or persisted["progress"],
            "runs": runs,
            "candidates": rank_hall_of_fame(candidates, 100),
            "archive_count": factor_mining_repo.archive_count(),
        }

    def cancel(self, campaign_id):
        persisted = factor_mining_repo.get_campaign(campaign_id)
        if not persisted:
            return {"campaign_id": campaign_id, "cancel_requested": False}
        current_run_id = (persisted.get("progress") or {}).get("current_run_id")
        if current_run_id:
            factor_mining_service.cancel(current_run_id)
        return {
            "campaign_id": campaign_id,
            "cancel_requested": task_manager.request_cancel(campaign_id),
        }

    def archive(self, limit=100, status=None):
        return factor_mining_repo.list_archive(limit=limit, status=status)


factor_mining_campaign_service = FactorMiningCampaignService()
