import datetime
import threading
import time
import uuid

from core.shared.infrastructure.async_task import TaskCancelled, task_manager


class DataRepairService:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_job_id = None
        self._active_scope = None
        self._previews = {}

    def latest_completed_trading_day(self):
        from core.data.collection_gate import GateCheckService
        return GateCheckService().get_latest_completed_trading_day()

    def resolve_repair_func(self, table):
        if table == "north_flow":
            return None
        if table == "stock_sectors":
            return self._repair_stock_sectors
        if table == "stock_notices":
            return self._repair_stock_notices
        if table == "stock_identity":
            return self._repair_stock_identity
        if table == "stock_index_membership":
            return self._repair_stock_index_membership
        if table == "stock_dividend_events":
            return self._repair_dividend_events
        if table == "raw_news":
            return self._repair_raw_news
        if table == "fundamental_data":
            return self._repair_fundamental
        if table in ("analyst_forecast_report_captures", "analyst_forecast_report_versions"):
            return self._repair_analyst_forecast_reports
        if table == "ths_daily_kline":
            return self._repair_ths_daily
        if table == "qlib_binary":
            return self._repair_qlib
        from core.data.integrity import CHECKS
        for check in CHECKS:
            if check["table"] == table:
                return check.get("repair")
        return None

    def preview_all(self):
        from core.data.collection_gate import GateCheckService, GateStatus, RULES

        latest_td = self.latest_completed_trading_day()
        service = GateCheckService()
        checks = service.check_all(force=True)
        targets = [
            rule.table for rule in RULES
            if checks[rule.table].status != GateStatus.FRESH
            and checks[rule.table].should_collect
            and self.resolve_repair_func(rule.table) is not None
        ]
        try:
            qlib_check = self._snapshot("qlib_binary", latest_td)
        except Exception as exc:
            qlib_check = {"status": "error", "reason": str(exc)[:300], "latest_date": None, "row_count": 0}
        if qlib_check["status"] != "fresh":
            targets.append("qlib_binary")
        rule_map = {rule.table: rule for rule in RULES}
        source_map = {
            "qlib_binary": "QlibDataBridge 增量同步",
            "stock_identity": "backend.service.stock_context.collect_stock_identity",
            "stock_index_membership": "backend.service.stock_context.collect_stock_index_membership",
            "stock_dividend_events": "backend.service.stock_context.collect_dividend_events",
        }
        items = []
        for table in targets:
            check = qlib_check if table == "qlib_binary" else checks.get(table)
            repair_func = self.resolve_repair_func(table)
            latest_date = check.get("latest_date") if table == "qlib_binary" else check.latest_in_db if check else None
            restart_required = table == "qlib_binary" and check.get("restart_required", False)
            items.append({
                "table": table,
                "desc": rule_map[table].desc if table in rule_map else "Qlib二进制数据",
                "status": check.get("status") if table == "qlib_binary" else check.status.value if check else "stale",
                "latest_date": str(latest_date)[:10] if latest_date else None,
                "row_count": check.get("row_count") if table == "qlib_binary" else check.row_count if check else None,
                "repair_scope": "等待重启后验证，不执行重复同步" if restart_required else "最近交易日检查及增量回补；不做全历史修复",
                "source": source_map.get(table, f"{repair_func.__module__}.{repair_func.__name__}" if repair_func else "未配置"),
                "write_target": "Qlib 二进制目录" if table == "qlib_binary" else f"正式表 {table}",
                "completion_check": "执行后重新运行对应数据质量门控",
                "downstream": "训练与回测会读取更新后的 Qlib 数据" if table == "qlib_binary" else "按数据表质量门控复核；不代表历史实验准入",
                "reason": check.get("reason") if table == "qlib_binary" else check.reason if check else None,
            })
        preview_id = f"repair-preview-{uuid.uuid4().hex[:10]}"
        self._previews[preview_id] = {"targets": targets, "expires_at": time.time() + 300}
        self._previews = {key: value for key, value in self._previews.items() if value["expires_at"] > time.time()}
        return {
            "preview_id": preview_id,
            "expires_in_seconds": 300,
            "trade_date": latest_td.isoformat(),
            "items": items,
            "target_count": len(items),
            "history_repair_included": False,
        }

    def start_all(self, preview_id=None):
        targets = None
        if preview_id:
            preview = self._previews.pop(preview_id, None)
            if not preview or preview["expires_at"] <= time.time():
                return {"status": "preview_expired", "message": "预览已过期，请重新预览修复范围"}
            targets = preview["targets"]
        return self._start(None, targets)

    def start_table(self, table):
        if self.resolve_repair_func(table) is None:
            return {"status": "not_supported", "message": f"表 {table} 暂无自动修复功能"}
        return self._start(table)

    def _start(self, table, targets=None):
        scope = (table, tuple(targets) if targets is not None else None)
        with self._lock:
            if self._active_job_id and task_manager.is_active(self._active_job_id):
                if self._active_scope == scope:
                    return {"job_id": self._active_job_id, "status": "running", "reused": True}
                return {"job_id": self._active_job_id, "status": "busy", "message": "已有范围不同的数据修复任务运行中，请等待其完成后重试"}
            job_id = f"repair{uuid.uuid4().hex[:6]}"
            self._active_job_id = job_id
            self._active_scope = scope
        try:
            task_manager.submit_with_id(
                job_id,
                lambda: self._run(job_id, table, targets),
                f"数据修复: {table or '全部异常表'}",
                pool="default",
            )
        except Exception:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                    self._active_scope = None
            raise
        return {"job_id": job_id, "status": "pending", "table": table}

    def _run(self, job_id, table, preview_targets=None):
        try:
            from core.data.collection_gate import GateCheckService, GateStatus, RULES

            latest_td = self.latest_completed_trading_day()
            service = GateCheckService()
            rule_map = {rule.table: rule for rule in RULES}
            if table:
                targets = [table]
            elif preview_targets is not None:
                targets = list(preview_targets)
            else:
                checks = service.check_all(force=True)
                targets = [
                    rule.table for rule in RULES
                    if checks[rule.table].status != GateStatus.FRESH
                    and checks[rule.table].should_collect
                    and self.resolve_repair_func(rule.table) is not None
                ]
                if self._snapshot("qlib_binary", latest_td)["status"] != "fresh":
                    targets.append("qlib_binary")

            results = []
            total = len(targets)
            if not targets:
                task_manager.update_progress(job_id, {
                    "stage": "completed", "current": 0, "total": 0,
                    "message": "所有数据均已完整，无需修复", "results": [],
                })
                return {"date": latest_td.isoformat(), "results": [], "all_pass": True}

            for index, target in enumerate(targets, 1):
                if task_manager.is_cancel_requested(job_id):
                    raise TaskCancelled(job_id)
                desc = rule_map[target].desc if target in rule_map else "Qlib二进制数据"
                before = self._snapshot(target, latest_td)
                task_manager.update_progress(job_id, {
                    "stage": "repairing", "current": index - 1, "total": total,
                    "table": target, "message": f"正在修复 {desc}", "results": results,
                })
                item = {"table": target, "desc": desc, "before": before}
                try:
                    if before.get("restart_required"):
                        item.update({"status": "partial", "after": before, "restart_required": True})
                        results.append(item)
                        continue
                    saved = self._call_with_heartbeat(
                        job_id,
                        self.resolve_repair_func(target),
                        latest_td.isoformat(),
                        {
                            "stage": "repairing", "current": index - 1, "total": total,
                            "table": target, "message": f"正在等待 {desc} 数据源返回",
                            "results": results,
                        },
                    )
                    after = self._snapshot(target, latest_td)
                    item.update({
                        "status": "completed" if after["status"] == "fresh" else "partial",
                        "saved": saved, "after": after,
                    })
                    if after.get("restart_required"):
                        item["restart_required"] = True
                except Exception as exc:
                    item.update({"status": "error", "error": str(exc)[:500]})
                results.append(item)
                task_manager.update_progress(job_id, {
                    "stage": "repairing", "current": index, "total": total,
                    "table": target, "message": f"已处理 {index}/{total} 项", "results": results,
                })

            all_pass = all(item["status"] == "completed" for item in results)
            restart_required = any(item.get("restart_required") for item in results)
            task_manager.update_progress(job_id, {
                "stage": "completed" if all_pass else "partial",
                "current": total, "total": total, "results": results,
                "message": "Qlib等待重启后验证；其他项目请查看结果" if restart_required else "全部修复完成" if all_pass else "修复完成，部分数据源仍未达标",
            })
            result = {"date": latest_td.isoformat(), "results": results, "all_pass": all_pass}
            if restart_required:
                result["restart_required"] = True
            if not all_pass:
                result["_task_status"] = "partial"
            return result
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                    self._active_scope = None

    @staticmethod
    def _call_with_heartbeat(job_id, repair_func, trade_date, progress):
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(30):
                task_manager.update_progress(job_id, progress)

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            return repair_func(trade_date)
        finally:
            stopped.set()
            worker.join(timeout=1)

    def _snapshot(self, table, latest_td):
        if table == "qlib_binary":
            from core.shared import config
            from core.adapters.qlib.bridge import QlibDataBridge
            from backend.service.qlib_access import run_with_qlib_read_access
            from core.shared.qlib_errors import QlibRestartRequired
            try:
                status = run_with_qlib_read_access(
                    lambda: QlibDataBridge(getattr(config, "QLIB_DATA_DIR", "data/qlib_data")).get_status()
                )
            except QlibRestartRequired as exc:
                return {
                    "status": "restart_required", "latest_date": None, "row_count": None,
                    "expected_min": 4000, "restart_required": True, "reason": str(exc),
                    "generation_id": exc.generation_id,
                    "current_generation_id": exc.current_generation_id,
                }
            last_date = status.get("last_date")
            fresh = bool(status.get("synced") and last_date and str(last_date)[:10] >= latest_td.isoformat())
            return {
                "status": "fresh" if fresh else "stale",
                "latest_date": last_date,
                "row_count": int(status.get("stocks") or 0),
                "expected_min": 4000,
            }

        from core.data.collection_gate import GateCheckService
        service = GateCheckService()
        result = service.check_table(table, force=True)
        return {
            "status": result.status.value,
            "latest_date": str(result.latest_in_db)[:10] if result.latest_in_db else None,
            "row_count": result.row_count,
            "expected_min": result.expected_min_rows,
            "reason": result.reason,
        }

    @staticmethod
    def _repair_stock_sectors(_trade_date=None):
        from backend.service.stock_context import collect_stock_concept_membership
        result = collect_stock_concept_membership()
        return result["current_relations"]

    @staticmethod
    def _repair_stock_notices(trade_date=None):
        from core.adapters.nlp.notice_collector import NoticeCollector
        date_str = str(trade_date or datetime.date.today()).replace("-", "")[:8]
        collector = NoticeCollector()
        return collector.save_to_db(collector.collect_full_market(date_str))

    @staticmethod
    def _repair_raw_news(_trade_date=None):
        from core.adapters.nlp.collector import SentimentCollector
        collector = SentimentCollector()
        items = collector.collect_market_snapshot()
        return collector.save_to_db(items) or 0

    @staticmethod
    def _repair_fundamental(_trade_date=None):
        from backend.service.fundamental_collection import fetch_and_save_fundamentals
        return fetch_and_save_fundamentals(max_workers=4)

    @staticmethod
    def _repair_analyst_forecast_reports(_trade_date=None):
        from backend.service.analyst_forecast_collection import collect_latest_analyst_forecast_reports
        return collect_latest_analyst_forecast_reports()

    @staticmethod
    def _repair_ths_daily(_trade_date=None):
        from core.data.ths_collector import THSCollector, _get_cookie, collect_cycle
        from core.shared.infrastructure.jobs import _get_ths_codes
        codes = _get_ths_codes()
        collector = THSCollector()
        collector.session.headers["Cookie"] = f"v={_get_cookie()}"
        return collect_cycle(codes, collector, "daily")

    @staticmethod
    def _repair_qlib(_trade_date=None):
        from core.shared import config
        from core.adapters.qlib.bridge import QlibDataBridge
        bridge = QlibDataBridge(getattr(config, "QLIB_DATA_DIR", "data/qlib_data"))
        return bridge.sync_incremental()

    @staticmethod
    def _repair_stock_identity(_trade_date=None):
        from backend.service.stock_context import collect_stock_identity, collect_stock_identity_industry_fallback
        identity = collect_stock_identity()
        industry_fallback = collect_stock_identity_industry_fallback()
        return {"identity_rows": identity, "industry_fallback": industry_fallback}

    @staticmethod
    def _repair_stock_index_membership(_trade_date=None):
        from backend.service.stock_context import collect_stock_index_membership
        return collect_stock_index_membership()

    @staticmethod
    def _repair_dividend_events(_trade_date=None):
        from backend.service.stock_context import collect_dividend_events
        return collect_dividend_events()


data_repair_service = DataRepairService()
