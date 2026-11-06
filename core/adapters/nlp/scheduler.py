"""情绪分析定时调度器——每日 08:45 / 12:50 / 20:00 自动采集（仅交易日）"""
import logging
import threading
import time
from datetime import datetime

from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

RUN_TIMES = [(8, 45), (12, 50), (20, 0)]


class SentimentScheduler:

    def __init__(self):
        self._registered = False
        self._last_run = None
        self._status = "stopped"
        self._next_run = None
        self._run_lock = threading.Lock()

    @property
    def running(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            meta = get_unified_scheduler().get_meta("sentiment-morning")
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
            "next_run": _next("sentiment-morning"),
            "status": self._status,
            "interval": f"每日 {', '.join(f'{h:02d}:{m:02d}' for h, m in RUN_TIMES)}",
            "schedule": [f"{h:02d}:{m:02d}" for h, m in RUN_TIMES],
        }

    def start(self):
        if self._registered:
            return

        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()

        sessions = [
            ("sentiment-morning", "45 8 * * mon-fri"),
            ("sentiment-noon", "50 12 * * mon-fri"),
            ("sentiment-evening", "0 20 * * mon-fri"),
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
        logger.info(f"情绪调度器已启动 (交易日 {'/'.join(f'{h:02d}:{m:02d}' for h, m in RUN_TIMES)})")

        try:
            from core.shared.trading_time import is_trading_day
            from core.data.collection_gate import GateCheckService
            now = datetime.now()
            gate = GateCheckService().check_table("raw_news", force=True)
            if is_trading_day(now.date()) and (now.hour, now.minute) >= (20, 0) and gate.should_collect:
                logger.warning(f"新闻数据晚间仍未完成，启动轻量补跑: {gate.reason}")
                thread_pool.default.submit(self._collect_global_index)
        except Exception as e:
            logger.warning(f"新闻启动补跑检查失败: {e}")

    def stop(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        for name in ["sentiment-morning", "sentiment-noon", "sentiment-evening"]:
            try:
                get_unified_scheduler().pause(name)
            except Exception:
                pass
        self._registered = False
        logger.info("情绪调度器已停止")

    def trigger(self):
        thread_pool.light.submit(self._run_once)

    def _run_once(self):
        if not self._run_lock.acquire(blocking=False):
            logger.warning("情绪采集任务仍在运行，跳过重复触发")
            return
        self._status = "running"
        start = time.time()
        try:
            stocks = self._get_tracked_stocks()
            logger.info(f"情绪采集: {len(stocks)} 只跟踪股票")
            if not stocks:
                self._status = "idle (无跟踪股票)"
                return

            from core.adapters.nlp.collector import SentimentCollector
            from core.adapters.nlp.analyzer import SentimentAnalyzer
            from core.adapters.nlp.factor import SentimentFactorCalculator
            from core.shared.infrastructure.thread_pool import thread_pool

            try:
                self._collect_global_index()
            except Exception:
                pass
            cailian_snapshot = SentimentCollector._fetch_cailian_all()
            analysis_date = datetime.now().strftime("%Y-%m-%d")

            _local = threading.local()

            def _process_stock(code):
                if not hasattr(_local, 'collector'):
                    _local.collector = SentimentCollector(cailian_cache=cailian_snapshot)
                    _local.analyzer = SentimentAnalyzer()
                try:
                    items = _local.collector.collect_single(code)
                    if not items:
                        return [], 0, 0
                    items = _local.collector.dedup(items)
                    fetched_count = len(items)
                    items = _local.collector.select_for_analysis(items, analysis_date)
                    if not items:
                        return [], fetched_count, fetched_count
                    analyzed = _local.analyzer.analyze(items)
                    _local.collector.save_to_db(analyzed)
                    return analyzed, fetched_count, fetched_count - len(items)
                except Exception as e:
                    logger.debug(f"采集异常({code}): {e}")
                    return [], 0, 0

            all_analyzed = []
            completed = 0
            total_news = 0
            fetched_news = 0
            skipped_existing = 0
            chunk_size = max(1, thread_pool.default._queue_size)
            total = len(stocks)

            for i in range(0, total, chunk_size):
                chunk = stocks[i:i + chunk_size]
                chunk_results = thread_pool.default.map_all(_process_stock, chunk)
                for result in chunk_results:
                    if isinstance(result, Exception):
                        continue
                    analyzed, fetched_count, skipped_count = result
                    fetched_news += fetched_count
                    skipped_existing += skipped_count
                    if analyzed:
                        all_analyzed.extend(analyzed)
                completed += len(chunk)
                elapsed = time.time() - start
                logger.info(
                    f"情绪采集进度: {completed}/{total} ({completed*100//total}%), "
                    f"待分析 {len(all_analyzed)}/{fetched_news}, 跳过历史重复 {skipped_existing}, {elapsed:.0f}s"
                )

            if all_analyzed:
                total_news = len(all_analyzed)
                calculator = SentimentFactorCalculator()
                by_code = {}
                for item in all_analyzed:
                    code = item.get("code", "")
                    by_code.setdefault(code, []).append(item)
                for code, code_items in by_code.items():
                    daily = calculator.calc_daily(code, code_items)
                    if daily:
                        calculator.save_daily(daily)

            elapsed = time.time() - start
            self._last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._status = (
                f"done ({len(stocks)} 只, 分析 {total_news}/{fetched_news} 条新闻, "
                f"跳过历史重复 {skipped_existing} 条, {elapsed:.1f}s)"
            )
            logger.info(f"情绪采集完成: {self._status}")

        except Exception as e:
            self._status = f"error: {e}"
            logger.exception(f"情绪采集失败: {e}")
        finally:
            self._run_lock.release()

    @staticmethod
    def _collect_global_index():
        from core.adapters.nlp.collector import SentimentCollector
        collector = SentimentCollector()
        items = collector.collect_market_snapshot()
        saved = collector.save_to_db(items)
        logger.info(f"全市场新闻索引采集完成: {saved} 条")
        return saved

    def _get_tracked_stocks(self):
        codes = []
        try:
            from core.data import Database
            if Database.is_available():
                rows = Database.fetchall(
                    "SELECT code, COUNT(*) as cnt FROM daily_quotes "
                    "GROUP BY code HAVING cnt >= 300 ORDER BY cnt DESC"
                )
                codes = [r["code"] for r in rows]
                logger.info(f"情绪采集范围: daily_quotes 池 {len(codes)} 只 (≥300条K线, 不设上限)")
        except Exception as e:
            logger.warning(f"daily_quotes查询失败: {e}")

        return codes


_sentiment_scheduler = None


def get_sentiment_scheduler() -> SentimentScheduler:
    global _sentiment_scheduler
    if _sentiment_scheduler is None:
        _sentiment_scheduler = SentimentScheduler()
    return _sentiment_scheduler
