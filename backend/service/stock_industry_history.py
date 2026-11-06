import csv
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import threading
import time
import uuid

from core.shared.infrastructure.async_task import task_manager


def repair_stock_industry_history_classification_standard(expected_snapshot_count, damaged_hex,
                                                          canonical_standard, canonical_row_hex):
    from core.data.repos.stock_industry_history import repair_classification_standard_metadata

    return repair_classification_standard_metadata(
        expected_snapshot_count=expected_snapshot_count,
        damaged_hex=damaged_hex,
        canonical_standard=canonical_standard,
        canonical_row_hex=canonical_row_hex,
    )


_lock = threading.Lock()
_active_job_id = None
_active_scope = None


def _mondays(start_date, end_date):
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    if start > end:
        raise ValueError("开始日期必须早于或等于结束日期")
    first = start + datetime.timedelta(days=(7 - start.weekday()) % 7)
    dates = []
    while first <= end:
        dates.append(first)
        first += datetime.timedelta(days=7)
    return dates


def _read_snapshot_file(path, expected_payload_hash):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = [{key: row.get(key, "") for key in ("ts_code", "symbol", "code_name", "industry", "industry_classification")} for row in rows]
    actual_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if actual_hash != expected_payload_hash:
        raise RuntimeError("BaoStock 原始快照文件与采集器返回的内容哈希不一致")
    codes = [row["ts_code"] for row in rows]
    if len(rows) < 2500 or len(codes) != len(set(codes)):
        raise RuntimeError(f"行业快照文件校验失败: rows={len(rows)}, unique_codes={len(set(codes))}")
    for row in rows:
        if row.get("industry") == "":
            row["industry"] = None
        if row.get("industry_classification") == "":
            row["industry_classification"] = None
    return rows


def start_stock_industry_history(start_date="2018-12-24", end_date=None, force=False):
    global _active_job_id, _active_scope
    from core.data.repos.stock_industry_history import ensure_stock_industry_history_tables

    if end_date is None:
        today = datetime.date.today()
        end_date = (today - datetime.timedelta(days=(today.weekday() + 7) % 7)).isoformat()
    dates = _mondays(start_date, end_date)
    if not dates:
        raise ValueError("所选范围不包含周一行业快照日")
    ensure_stock_industry_history_tables()
    scope = (str(dates[0]), str(dates[-1]), bool(force))
    with _lock:
        if _active_job_id and task_manager.is_active(_active_job_id):
            if _active_scope == scope:
                return {"job_id": _active_job_id, "status": "running", "reused": True}
            return {"job_id": _active_job_id, "status": "busy", "message": "已有行业历史采集任务运行中"}
        job_id = f"industry{uuid.uuid4().hex[:8]}"
        _active_job_id = job_id
        _active_scope = scope

    def run(progress_cb):
        global _active_job_id, _active_scope
        from core.data.repos.stock_industry_history import (
            get_completed_query_dates,
            get_industry_history_counts,
            save_industry_history_failure,
            save_industry_history_snapshot,
        )

        root = pathlib.Path(__file__).resolve().parents[2]
        artifact_dir = root / "data" / "historical_repair" / "industry_snapshots"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        completed = get_completed_query_dates(dates[0], dates[-1])
        pending = [day for day in dates if force or day.isoformat() not in completed]
        failures = []
        saved = 0
        try:
            for index, query_day in enumerate(pending, 1):
                task_manager.check_cancelled(job_id)
                query_date = query_day.isoformat()
                final_error = None
                for attempt in range(1, 4):
                    capture_path = artifact_dir / f"staging_{query_day:%Y%m%d}_{uuid.uuid4().hex[:8]}.csv"
                    try:
                        process = subprocess.run(
                            [sys.executable, "-m", "core.data.sources.baostock_industry_worker",
                             query_date, str(capture_path)],
                            cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=150,
                        )
                        if process.returncode != 0:
                            raise RuntimeError((process.stderr or process.stdout or f"worker exit={process.returncode}")[-1200:])
                        worker_result = json.loads(process.stdout.strip().splitlines()[-1])
                        file_hash = hashlib.sha256()
                        with open(capture_path, "rb") as captured:
                            for chunk in iter(lambda: captured.read(1024 * 1024), b""):
                                file_hash.update(chunk)
                        if file_hash.hexdigest() != worker_result.get("file_sha256"):
                            raise RuntimeError("BaoStock 原始文件哈希与采集器返回值不一致")
                        rows = _read_snapshot_file(capture_path, worker_result["payload_sha256"])
                        source_update_date = datetime.date.fromisoformat(worker_result["source_update_date"])
                        if worker_result["query_date"] != query_date or source_update_date > query_day:
                            raise RuntimeError("来源查询日期或更新日期越界")
                        available_after_date = source_update_date + datetime.timedelta(days=1)
                        observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                        snapshot_id = hashlib.sha256(
                            f"baostock.query_stock_industry|{query_date}|{source_update_date}|{worker_result['payload_sha256']}".encode("utf-8")
                        ).hexdigest()[:32]
                        archive_path = artifact_dir / f"stock_industry_{query_day:%Y%m%d}_{worker_result['payload_sha256'][:12]}.csv"
                        if archive_path.exists():
                            capture_path.unlink(missing_ok=True)
                        else:
                            capture_path.replace(archive_path)
                        artifact_path = archive_path.relative_to(root).as_posix()
                        artifact_sha256 = worker_result["file_sha256"]
                        snapshot = {
                            "snapshot_id": snapshot_id,
                            "query_date": query_date,
                            "source_update_date": source_update_date.isoformat(),
                            "available_after_date": available_after_date.isoformat(),
                            "observed_at": observed_at,
                            "source_name": "baostock.query_stock_industry",
                            "classification_standard": worker_result.get("classification_standard") or None,
                            "row_count": len(rows),
                            "unique_code_count": len({row["ts_code"] for row in rows}),
                            "payload_sha256": worker_result["payload_sha256"],
                            "artifact_sha256": artifact_sha256,
                            "artifact_path": artifact_path,
                        }
                        save_industry_history_snapshot(snapshot, rows)
                        saved += 1
                        final_error = None
                        break
                    except Exception as exc:
                        final_error = str(exc)
                        if attempt < 3:
                            time.sleep(attempt * 2)
                    finally:
                        if capture_path.exists() and final_error:
                            capture_path.unlink(missing_ok=True)
                if final_error:
                    failure_id = hashlib.sha256(f"industry-failed|{query_date}|{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:32]
                    observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                    save_industry_history_failure(failure_id, query_date, observed_at, final_error)
                    failures.append({"query_date": query_date, "error": final_error})
                progress_cb({
                    "stage": "collecting_industry_history",
                    "current": index,
                    "total": len(pending),
                    "completed_total": len(completed) + saved,
                    "requested_total": len(dates),
                    "failed_total": len(failures),
                    "message": f"行业历史快照 {query_date} 完成：成功 {saved}，失败 {len(failures)}",
                })
            summary = get_industry_history_counts(dates[0], dates[-1])
            return {
                "status": "partial" if failures else "complete",
                "_task_status": "partial" if failures else "completed",
                "requested_snapshots": len(dates),
                "already_completed": len(completed),
                "newly_saved": saved,
                "failed": failures,
                "summary": summary,
                "pit_status": "not_certified",
                "availability_rule": "source_update_date_plus_1_calendar_day",
            }
        finally:
            with _lock:
                if _active_job_id == job_id:
                    _active_job_id = None
                    _active_scope = None

    try:
        task_manager.submit_with_id(job_id, run, "采集 BaoStock 历史行业快照",
                                    progress_cb=lambda progress: task_manager.update_progress(job_id, progress),
                                    pool="default")
    except Exception:
        with _lock:
            if _active_job_id == job_id:
                _active_job_id = None
                _active_scope = None
        raise
    return {"job_id": job_id, "status": "pending", "start_date": str(dates[0]), "end_date": str(dates[-1]),
            "requested_snapshots": len(dates), "force": bool(force)}


def get_stock_industry_history_status(start_date=None, end_date=None, ensure=True):
    from core.data.repos.stock_industry_history import (
        ensure_stock_industry_history_tables,
        get_completed_query_dates,
        get_industry_history_counts,
        get_industry_history_failures,
    )

    if ensure:
        ensure_stock_industry_history_tables()
    if start_date is None:
        start_date = "2018-12-24"
    if end_date is None:
        today = datetime.date.today()
        end_date = (today - datetime.timedelta(days=(today.weekday() + 7) % 7)).isoformat()
    dates = _mondays(start_date, end_date)
    summary = get_industry_history_counts(start_date, end_date)
    completed_dates = get_completed_query_dates(dates[0], dates[-1]) if dates else set()
    missing_dates = [day.isoformat() for day in dates if day.isoformat() not in completed_dates]
    failures = get_industry_history_failures(dates[0], dates[-1]) if dates else {}
    if summary.get("first_query_date") is not None:
        for key in ("first_query_date", "last_query_date", "first_source_update_date", "last_source_update_date"):
            if summary.get(key) is not None:
                summary[key] = summary[key].isoformat()
    if failures.get("last_failure_at") is not None:
        failures["last_failure_at"] = failures["last_failure_at"].isoformat()
    return {
        "status": "complete" if dates and not missing_dates else "partial" if int(summary.get("snapshots") or 0) else "missing",
        "summary": summary,
        "failures": failures,
        "expected_snapshots": len(dates),
        "missing_snapshot_count": len(missing_dates),
        "missing_snapshot_dates_sample": missing_dates[:12],
        "pit_status": "not_certified",
        "availability_rule": "source_update_date_plus_1_calendar_day",
        "note": "BaoStock 更新频率为每周一；精确发布时间未验证，采集快照不得单独作为严格 PIT 认证。",
    }
