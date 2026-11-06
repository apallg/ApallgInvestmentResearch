"""公告采集定时调度器——交易日 09:00 / 15:05 / 20:30 全市场批量采集"""
import logging
import time
from datetime import datetime

from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

RUN_TIMES = [(9, 0), (15, 5), (20, 30)]


class NoticeScheduler:

    def __init__(self):
        self._registered = False
        self._last_run = None
        self._status = "stopped"
        self._next_run = None

    @property
    def running(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            meta = get_unified_scheduler().get_meta("notice-morning")
            return meta is not None and meta.get("scheduled", False)
        except Exception:
            return False

    @property
    def status(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            mgr = get_unified_scheduler()
        except Exception:
            mgr = None

        def _next(name):
            if mgr is None:
                return None
            m = mgr.get_meta(name)
            return m.get("next_run") if m else None

        return {
            "running": self.running,
            "last_run": self._last_run,
            "next_run": _next("notice-morning"),
            "status": self._status,
            "schedule": [f"{h:02d}:{m:02d}" for h, m in RUN_TIMES],
        }

    def start(self):
        if self._registered:
            return

        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()

        sessions = [
            ("notice-morning", "0 9 * * mon-fri"),
            ("notice-afternoon", "5 15 * * mon-fri"),
            ("notice-evening", "30 20 * * mon-fri"),
        ]

        for name, cron in sessions:
            meta = mgr.get_meta(name)
            if meta is None:
                mgr.register(name, self._run_once, cron, trading_day_only=True)
            elif not meta.get("scheduled"):
                mgr.resume(name)

        if not mgr._started:
            mgr.start()

        self._registered = True
        logger.info(f"公告调度器已启动 (交易日 {'/'.join(f'{h:02d}:{m:02d}' for h, m in RUN_TIMES)})")

        try:
            from core.shared.trading_time import is_trading_day
            from core.data.collection_gate import GateCheckService
            now = datetime.now()
            gate = GateCheckService().check_table("stock_notices", force=True)
            if is_trading_day(now.date()) and (now.hour, now.minute) >= (15, 5) and gate.should_collect:
                logger.warning(f"公告数据收盘后仍未完成，启动补跑: {gate.reason}")
                self.trigger()
        except Exception as e:
            logger.warning(f"公告启动补跑检查失败: {e}")

    def stop(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        for name in ["notice-morning", "notice-afternoon", "notice-evening"]:
            try:
                get_unified_scheduler().pause(name)
            except Exception:
                pass
        self._registered = False
        logger.info("公告调度器已停止")

    def trigger(self):
        thread_pool.light.submit(self._run_once)

    def _run_once(self):
        self._status = "running"
        start = time.time()
        try:
            today = datetime.now().strftime("%Y%m%d")
            logger.info(f"公告采集开始: {today}")

            from core.adapters.nlp.notice_collector import NoticeCollector
            collector = NoticeCollector()
            items = collector.collect_full_market(today)

            if items:
                saved = collector.save_to_db(items)
            else:
                saved = 0

            elapsed = time.time() - start
            self._last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._status = f"done ({saved} 条公告, {elapsed:.1f}s)"
            logger.info(f"公告采集完成: {self._status}")

        except Exception as e:
            self._status = f"error: {e}"
            logger.exception(f"公告采集失败: {e}")


_notice_scheduler = None


def get_notice_scheduler() -> NoticeScheduler:
    global _notice_scheduler
    if _notice_scheduler is None:
        _notice_scheduler = NoticeScheduler()
    return _notice_scheduler
