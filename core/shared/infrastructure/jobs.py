"""交易日定时扫描调度器 + 每日数据采集调度器"""
import logging
import time
import datetime
import threading

from core.shared import config

from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

MORNING_TIME = getattr(config, "SCAN_MORNING_TIME", "09:40")
MIDDAY_TIME = getattr(config, "SCAN_MIDDAY_TIME", "13:00")
AFTERNOON_TIME = getattr(config, "SCAN_AFTERNOON_TIME", "14:55")
DEFAULT_THRESHOLD = getattr(config, "SCAN_DEFAULT_THRESHOLD", 7)

SESSION_AM = "早盘"
SESSION_MD = "午盘"
SESSION_PM = "尾盘"
SESSION_MANUAL = "手动"


def _parse_time(s):
    h, m = map(int, s.split(":"))
    return datetime.time(h, m)


from core.shared.trading_time import is_trading_day


class TradingDayScheduler:
    def __init__(self, scan_fn=None):
        self._registered = False
        self._busy = False
        self._morning_t = _parse_time(MORNING_TIME)
        self._midday_t = _parse_time(MIDDAY_TIME)
        self._afternoon_t = _parse_time(AFTERNOON_TIME)
        self.morning_time = MORNING_TIME
        self.midday_time = MIDDAY_TIME
        self.afternoon_time = AFTERNOON_TIME
        self.default_threshold = DEFAULT_THRESHOLD
        self.last_scan = None
        self.last_result = None
        self._scan_fn = scan_fn

    def set_scan_fn(self, fn):
        self._scan_fn = fn

    @property
    def running(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            meta = get_unified_scheduler().get_meta("risk-scan-morning")
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

        def _m(name):
            if mgr is None:
                return {}
            return mgr.get_meta(name) or {}

        meta_m = _m("risk-scan-morning")
        meta_md = _m("risk-scan-midday")
        meta_a = _m("risk-scan-afternoon")

        return {
            "running": self.running,
            "morning_time": self.morning_time,
            "midday_time": self.midday_time,
            "afternoon_time": self.afternoon_time,
            "default_threshold": self.default_threshold,
            "last_scan": self.last_scan,
            "last_result": self.last_result,
            "next_morning": meta_m.get("next_run"),
            "next_midday": meta_md.get("next_run"),
            "next_afternoon": meta_a.get("next_run"),
        }

    def start(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()

        sessions = [
            (SESSION_AM, "risk-scan-morning", f"{self._morning_t.minute} {self._morning_t.hour} * * mon-fri"),
            (SESSION_MD, "risk-scan-midday", f"{self._midday_t.minute} {self._midday_t.hour} * * mon-fri"),
            (SESSION_PM, "risk-scan-afternoon", f"{self._afternoon_t.minute} {self._afternoon_t.hour} * * mon-fri"),
        ]

        for session, name, cron in sessions:
            meta = mgr.get_meta(name)
            if meta is None:
                def _make_scan(s=session):
                    self._run_scan(s)
                mgr.register(name, _make_scan, cron, trading_day_only=True)
            elif not meta.get("scheduled"):
                mgr.resume(name)

        if not mgr._started:
            mgr.start()

        self._registered = True
        logger.info(f"交易日定时扫描已启动 {SESSION_AM}{self.morning_time} {SESSION_MD}{self.midday_time} {SESSION_PM}{self.afternoon_time} 阈值{self.default_threshold}")

    def stop(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()
        for name in ["risk-scan-morning", "risk-scan-midday", "risk-scan-afternoon"]:
            try:
                mgr.pause(name)
            except Exception:
                pass
        self._registered = False
        logger.info("交易日定时扫描已停止")

    def trigger(self):
        if self._busy:
            logger.warning("上一次扫描尚未完成，跳过本次手动触发")
            return
        thread_pool.light.submit(lambda: self._run_scan(SESSION_MANUAL))

    def _run_scan(self, session_name):
        if self._busy:
            logger.warning(f"上一次扫描尚未完成, 跳过 {session_name}")
            return
        self._busy = True
        try:
            scan_fn = self._scan_fn
            if scan_fn is None:
                self.last_result = f"{session_name}: 跳过 (未注册扫描函数)"
                logger.warning("风险扫描函数未注册, 跳过")
                return
            results = scan_fn(threshold=self.default_threshold, session_name=session_name)
            self.last_scan = datetime.datetime.now().isoformat()
            if isinstance(results, dict):
                self.last_result = (f"{session_name}: 应扫{results['expected']}只，已扫{results['scanned']}只，"
                                    f"未知{len(results['unknown'])}只，动态价位不可用{len(results.get('dynamic_unknown', []))}只，"
                                    f"触发{len(results['triggered'])}条")
            else:
                self.last_result = f"{session_name}: {len(results)}条触发"
            logger.info(f"定时扫描 [{session_name}] 完成: {self.last_result}")
        except Exception as e:
            self.last_result = f"{session_name}: 失败 - {e}"
            logger.error(f"定时扫描 [{session_name}] 失败: {e}")
        finally:
            self._busy = False


_scheduler = None


def get_scheduler(scan_fn=None) -> TradingDayScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TradingDayScheduler(scan_fn=scan_fn)
    elif scan_fn is not None:
        _scheduler.set_scan_fn(scan_fn)
    return _scheduler


class DataCollectionScheduler:

    def __init__(self):
        self._registered = False
        self._busy = False
        self.collect_time = getattr(config, "COLLECT_SCHEDULE_TIME", "18:30")
        self._collect_t = _parse_time(self.collect_time)
        self.stock_pool = getattr(config, "COLLECT_STOCK_POOL", "all")
        self.data_source = getattr(config, "COLLECT_DATA_SOURCE", "tencent")
        self.thread_count = getattr(config, "COLLECT_THREAD_COUNT", 16)
        self.last_collection = None
        self.last_result = None
        self.collection_count = 0
        self._collect_day_cache = {}
        self._task_registry = {}
        self._concept_sector_collector = None
        self._qlib_read_fn = None

    def set_qlib_read_fn(self, fn):
        self._qlib_read_fn = fn

    def register_task(self, name, fn):
        """注册采集子任务。fn 签名为 (run_date: date) -> result

        注意：仅在启动阶段调用，非线程安全。不要在调度器运行期间动态注册。
        """
        self._task_registry[name] = fn

    def register_concept_sector_collector(self, fn):
        self._concept_sector_collector = fn

    @property
    def running(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            meta = get_unified_scheduler().get_meta("data-collection")
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

        meta = mgr.get_meta("data-collection") if mgr else {}
        s = {
            "running": self.running,
            "busy": self._busy or (meta.get("running", False) if meta else False),
            "collect_time": self.collect_time,
            "stock_pool": self.stock_pool,
            "data_source": self.data_source,
            "thread_count": self.thread_count,
            "last_collection": self.last_collection,
            "last_result": self.last_result,
            "next_collection": meta.get("next_run") if meta else None,
            "collection_count": self.collection_count,
        }
        lr = self.last_result
        if isinstance(lr, dict):
            s["last_north_flow"] = lr.get("north_flow", 0)
            s["last_capital_flow"] = lr.get("capital_flow", 0)
        return s

    def start(self):
        if not getattr(config, "COLLECT_ENABLED", True) or not getattr(config, "MYSQL_ENABLED", False):
            logger.info("每日自动采集未启用 (COLLECT_ENABLED=false 或 MySQL 未配置)")
            return
        if self._registered:
            return

        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()
        cron = f"{self._collect_t.minute} {self._collect_t.hour} * * mon-fri"

        def _do():
            self._do_collection(source="scheduled")

        meta = mgr.get_meta("data-collection")
        if meta is None:
            mgr.register("data-collection", _do, cron, trading_day_only=True)
        elif not meta.get("scheduled"):
            mgr.resume("data-collection")

        if not mgr._started:
            mgr.start()

        self._registered = True
        logger.info(f"每日自动采集已启动 (收盘后 {self.collect_time}, 池={self.stock_pool}, 源={self.data_source})")

        try:
            now = datetime.datetime.now()
            if now.time() >= self._collect_t and is_trading_day(now.date()):
                from core.data.collection_gate import get_gate_service
                daily_gate = get_gate_service().check_table("daily_quotes")
                if daily_gate and daily_gate.should_collect:
                    logger.warning(f"检测到收盘后日K未完成，启动补跑: {daily_gate.reason}")
                    mgr.trigger("data-collection")
        except Exception as e:
            logger.warning(f"启动补跑检查失败: {e}")

        # ── 概念板块定时采集（每周一次，默认周五17:00）──
        if getattr(config, "COLLECT_CONCEPT_SECTORS_ENABLED", False):
            concept_day = str(getattr(config, "COLLECT_CONCEPT_SECTORS_DAY", "fri"))
            concept_day_cron = {
                "1": "mon", "2": "tue", "3": "wed", "4": "thu",
                "5": "fri", "6": "sat", "7": "sun",
            }.get(concept_day, concept_day)
            concept_time = getattr(config, "COLLECT_CONCEPT_SECTORS_TIME", "17:00")
            ct = _parse_time(concept_time)

            def _do_concept():
                if self._concept_sector_collector:
                    self._concept_sector_collector()
                else:
                    self._collect_concept_sectors()

            cm = mgr.get_meta("concept-sector-collection")
            if cm is None:
                mgr.register(
                    "concept-sector-collection", _do_concept,
                    f"{ct.minute} {ct.hour} * * {concept_day_cron}",
                    trading_day_only=True)
                logger.info(f"概念板块采集已启动 (周{concept_day} {concept_time})")
            elif not cm.get("scheduled"):
                mgr.resume("concept-sector-collection")

    def stop(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        try:
            get_unified_scheduler().pause("data-collection")
        except Exception:
            pass
        self._registered = False
        logger.info("每日自动采集已停止")

    def trigger(self):
        thread_pool.light.submit(self._do_collection, source=SESSION_MANUAL)

    def _do_collection(self, source="scheduled"):
        if self._busy:
            logger.warning("上一次采集尚未完成，跳过本次")
            return
        self._busy = True
        start_time = datetime.datetime.now()
        run_date = start_time.date()
        result = {
            "source": source,
            "data_source": self.data_source,
            "stock_pool": self.stock_pool,
            "start_time": start_time.isoformat(),
        }
        try:
            from scripts.fast_collect import get_all_codes, collect_incremental
            codes = get_all_codes(self.stock_pool)
            if self.stock_pool != "all":
                try:
                    from core.data.sources.tushare import get_all_stock_codes
                    active = set(get_all_stock_codes())
                    before = len(codes)
                    codes = [c for c in codes if str(c) in active]
                    if before != len(codes):
                        logger.info(f"退市过滤: {before} → {len(codes)}")
                except Exception:
                    pass
            total_codes = len(codes)
            logger.info(f"开始采集: {total_codes} 只股票, 源={self.data_source}, 池={self.stock_pool}")

            # ── 阶段0: 前置门控检查 ──
            gate_results = {}
            try:
                from core.data.collection_gate import get_gate_service
                gate = get_gate_service()
                gate_results = gate.check_all()
                gate.log_summary(gate_results)
                result["gate"] = {
                    t: {"status": r.status.value, "reason": r.reason}
                    for t, r in gate_results.items()
                }
            except Exception as _gate_err:
                logger.warning(f"门控检查失败, 回退为全量采集: {_gate_err}")
                result["gate_error"] = str(_gate_err)

            def _gate_ok(table):
                """检查某表是否需要采集: 门控失败/缺失时默认 True (安全回退)"""
                gr = gate_results.get(table)
                return gr.should_collect if gr else True

            kline_needed = _gate_ok("daily_quotes")

            if kline_needed:
                saved = collect_incremental(
                    codes,
                    source=self.data_source,
                    threads=self.thread_count,
                    delay=getattr(config, "COLLECT_DELAY_SECONDS", 0.1),
                    lookback_days=getattr(config, "COLLECT_LOOKBACK_DAYS", 3),
                )
            else:
                logger.info(f"[门控] 日K线数据已是最新 ({gate_results.get('daily_quotes').latest_in_db if gate_results.get('daily_quotes') else '?'}), 跳过采集")
                saved = 0

            result["primary_saved"] = saved
            try:
                from core.data import Database
                from core.shared.trading_time import get_latest_trading_day
                latest_td = get_latest_trading_day(run_date)
                expected_rows = getattr(config, "COLLECT_MIN_DAILY_ROWS", 4800)
                daily_row = Database.fetchone(
                    "SELECT COUNT(DISTINCT code) AS cnt FROM daily_quotes WHERE trade_date=%s "
                    "AND open IS NOT NULL AND close IS NOT NULL",
                    (latest_td,),
                )
                daily_count = int(daily_row.get("cnt") or 0) if daily_row else 0
                fallback_saved = 0
                if daily_count < expected_rows and self.data_source != "baostock":
                    logger.warning(
                        f"日K主源采集后仍不完整: {daily_count}/{expected_rows}, "
                        "切换 BaoStock 补采缺失股票"
                    )
                    fallback_saved = collect_incremental(
                        codes,
                        source="baostock",
                        threads=1,
                        delay=getattr(config, "COLLECT_DELAY_SECONDS", 0.1),
                        lookback_days=getattr(config, "COLLECT_LOOKBACK_DAYS", 3),
                        latest_trading_day=latest_td,
                    )
                    saved += fallback_saved
                    daily_row = Database.fetchone(
                        "SELECT COUNT(DISTINCT code) AS cnt FROM daily_quotes WHERE trade_date=%s "
                        "AND open IS NOT NULL AND close IS NOT NULL",
                        (latest_td,),
                    )
                    daily_count = int(daily_row.get("cnt") or 0) if daily_row else 0
                result["fallback_saved"] = fallback_saved
                result["kline_trade_date"] = latest_td.isoformat()
                result["kline_row_count"] = daily_count
                result["kline_expected_rows"] = expected_rows
                result["kline_complete"] = daily_count >= expected_rows
            except Exception as e:
                logger.warning(f"日K完整性补采失败: {e}")
                result["kline_complete"] = False
                result["kline_error"] = str(e)

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            result["total_codes"] = total_codes
            result["saved"] = saved
            result["elapsed_seconds"] = elapsed

            auto_sync_qlib = getattr(config, "COLLECT_AUTO_SYNC_QLIB", True)
            qlib_bridge = None
            kline_complete = bool(result.get("kline_complete"))
            qlib_sync_needed = saved > 0 and kline_complete
            if auto_sync_qlib and not qlib_sync_needed:
                if not kline_complete:
                    logger.warning("日K未达到完整性阈值，跳过 Qlib 日历同步")
                else:
                    try:
                        from core.adapters.qlib.bridge import QlibDataBridge
                        qlib_bridge = QlibDataBridge(config.QLIB_DATA_DIR)
                        qlib_status = qlib_bridge.get_status()
                        qlib_last_date = qlib_status.get("last_date")
                        db_last_date = result.get("kline_trade_date")
                        qlib_sync_needed = bool(
                            not qlib_status.get("synced")
                            or not qlib_last_date
                            or (db_last_date and qlib_last_date < db_last_date)
                        )
                        if qlib_sync_needed:
                            logger.info(
                                f"Qlib 日历落后数据库 ({qlib_last_date or '无'} < "
                                f"{db_last_date or '未知'}), 自动补同步"
                            )
                    except Exception as e:
                        logger.warning(f"Qlib 新鲜度检查失败, 保守触发同步: {e}")
                        qlib_sync_needed = True

            if auto_sync_qlib and qlib_sync_needed:
                logger.info("自动触发 Qlib 增量同步 (后台)...")
                def _qlib_sync():
                    try:
                        from core.adapters.qlib.bridge import QlibDataBridge
                        bridge = qlib_bridge or QlibDataBridge(config.QLIB_DATA_DIR)
                        sync_result = bridge.sync_incremental()
                        logger.info(f"Qlib 同步完成: +{sync_result.get('new_dates',0)} 交易日, +{sync_result.get('new_stocks',0)} 股票")
                        if sync_result.get("new_dates", 0) > 0:
                            from core.adapters.qlib.config import reload_calendar
                            reload_calendar()
                        return sync_result
                    except Exception as e:
                        logger.error(f"Qlib 同步失败: {e}")
                        return None
                qlib_future = thread_pool.default.submit(_qlib_sync)
            else:
                qlib_future = None

            # ── 并行执行所有独立子任务 ──
            subtask_futures = {}

            def _north_summary():
                if not getattr(config, "COLLECT_NORTH_FLOW_ENABLED", True):
                    return 0
                try:
                    from core.data.sources.tushare import get_pro
                    from core.data import Database
                    pro = get_pro()
                    if not pro:
                        logger.info("[北向] Tushare 不可用")
                        return 0
                    Database.execute("""
                        CREATE TABLE IF NOT EXISTS `north_flow_summary` (
                          id INT AUTO_INCREMENT PRIMARY KEY,
                          trade_date DATE NOT NULL UNIQUE,
                          north_net DECIMAL(16,2) DEFAULT NULL,
                          north_buy DECIMAL(16,2) DEFAULT NULL,
                          south_net DECIMAL(16,2) DEFAULT NULL,
                          hgt_net DECIMAL(16,2) DEFAULT NULL,
                          sgt_net DECIMAL(16,2) DEFAULT NULL,
                          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """)
                    ds = run_date.strftime("%Y%m%d")
                    df = pro.moneyflow_hsgt(trade_date=ds)
                    if df is not None and not df.empty:
                        r = df.iloc[0]
                        Database.execute(
                            "INSERT INTO north_flow_summary (trade_date, north_net, north_buy, south_net, hgt_net, sgt_net) "
                            "VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                            "north_net=VALUES(north_net), north_buy=VALUES(north_buy), "
                            "south_net=VALUES(south_net), hgt_net=VALUES(hgt_net), sgt_net=VALUES(sgt_net)",
                            (run_date,
                             float(r.get("north_money", 0) or 0),
                             float(r.get("hgt", 0) or 0) + float(r.get("sgt", 0) or 0),
                             float(r.get("south_money", 0) or 0),
                             float(r.get("hgt", 0) or 0),
                             float(r.get("sgt", 0) or 0)),
                        )
                        logger.info(f"北向资金汇总: {r.get('north_money', 0)}亿")
                        return 1
                    return 0
                except Exception as e:
                    logger.warning(f"北向资金采集失败: {e}")
                    return 0
            # 北向资金 — 交易所2024-08起停发，永久跳过
            if False:  # was: _gate_ok("north_flow_summary")
                subtask_futures["north_summary"] = thread_pool.default.submit(_north_summary)
            else:
                logger.info("[门控] 北向资金汇总已是最新, 跳过")

            def _north_detail():
                if not getattr(config, "COLLECT_NORTH_FLOW_ENABLED", True):
                    return 0
                task_fn = self._task_registry.get("north_detail")
                if task_fn is None:
                    logger.info("北向资金个股采集未注册, 跳过")
                    return 0
                try:
                    nf = task_fn(run_date)
                    if nf:
                        logger.info(f"北向资金个股采集: {nf} 只")
                    return nf or 0
                except Exception as e:
                    logger.warning(f"北向资金个股采集失败: {e}")
                    return 0
            if _gate_ok("north_flow"):
                subtask_futures["north_detail"] = thread_pool.default.submit(_north_detail)
            else:
                logger.info("[门控] 北向资金个股已是最新, 跳过")

            def _capital_flow():
                if not getattr(config, "COLLECT_CAPITAL_FLOW_ENABLED", True):
                    return 0
                task_fn = self._task_registry.get("capital_flow")
                if task_fn is None:
                    logger.info("资金流向采集未注册, 跳过")
                    return 0
                try:
                    c = task_fn(run_date)
                    logger.info(f"资金流向采集: {c} 只")
                    return c
                except Exception as e:
                    logger.warning(f"资金流向采集失败: {e}")
                    return 0
            if _gate_ok("capital_flow_daily"):
                subtask_futures["capital_flow"] = thread_pool.default.submit(_capital_flow)
            else:
                logger.info("[门控] 主力资金流向已是最新, 跳过")

            def _margin_daily():
                task_fn = self._task_registry.get("margin_daily")
                if task_fn is None:
                    logger.info("融资融券采集未注册, 跳过")
                    return 0
                try:
                    n = task_fn(run_date)
                    logger.info(f"融资融券采集: {n} 条")
                    return n or 0
                except Exception as e:
                    logger.warning(f"融资融券采集失败: {e}")
                    return 0
            if _gate_ok("margin_daily"):
                subtask_futures["margin_daily"] = thread_pool.default.submit(_margin_daily)

            def _lhb_daily():
                task_fn = self._task_registry.get("lhb_daily")
                if task_fn is None:
                    logger.info("龙虎榜采集未注册, 跳过")
                    return 0
                try:
                    n = task_fn(run_date)
                    logger.info(f"龙虎榜采集: {n} 条")
                    return n or 0
                except Exception as e:
                    logger.warning(f"龙虎榜采集失败: {e}")
                    return 0
            if _gate_ok("lhb_daily"):
                subtask_futures["lhb_daily"] = thread_pool.default.submit(_lhb_daily)

            def _shareholder_quarterly():
                task_fn = self._task_registry.get("shareholder_quarterly")
                if task_fn is None:
                    logger.info("股东户数采集未注册, 跳过")
                    return 0
                try:
                    n = task_fn(run_date)
                    logger.info(f"股东户数采集: {n} 条")
                    return n or 0
                except Exception as e:
                    logger.warning(f"股东户数采集失败: {e}")
                    return 0
            if _gate_ok("shareholder_quarterly"):
                subtask_futures["shareholder_quarterly"] = thread_pool.default.submit(_shareholder_quarterly)

            def _stock_context():
                if not getattr(config, "COLLECT_STOCK_CONTEXT_ENABLED", True):
                    return {}
                task_fn = self._task_registry.get("stock_context")
                if task_fn is None:
                    logger.info("证券身份/行业/分红快照采集未注册, 跳过")
                    return {}
                try:
                    result = task_fn(run_date)
                    logger.info(f"证券身份/行业/分红快照采集完成: {result}")
                    return result
                except Exception as e:
                    logger.warning(f"证券身份/行业/分红快照采集失败: {e}")
                    return {}
            subtask_futures["stock_context"] = thread_pool.default.submit(_stock_context)

            def _fundamental():
                if not (getattr(config, "COLLECT_FUNDAMENTAL_ENABLED", True) and self._should_collect("fundamental_data", 7)):
                    return {"saved": 0}
                task_fn = self._task_registry.get("fundamental")
                if task_fn is None:
                    logger.info("基本面采集未注册, 跳过")
                    return {"saved": 0}
                try:
                    fr = task_fn(run_date)
                    f = fr.get("saved", 0)
                    logger.info(f"基本面采集: {f} 条 (Tushare={fr.get('tushare',0)} akshare={fr.get('akshare',0)})")
                    return fr
                except Exception as e:
                    logger.warning(f"基本面采集失败: {e}")
                    return {"saved": 0}
            if _gate_ok("fundamental_data"):
                subtask_futures["fundamental"] = thread_pool.default.submit(_fundamental)
            else:
                logger.info("[门控] 基本面数据已是最新, 跳过")

            def _forecast():
                if not (getattr(config, "COLLECT_FORECAST_ENABLED", True) and self._should_collect("forecast_data", 7)):
                    return 0
                try:
                    from scripts.tushare_collect import collect_forecast
                    collect_forecast()
                    logger.info("一致预期采集完成")
                    return 1
                except Exception as e:
                    logger.warning(f"一致预期采集失败: {e}")
                    return 0
            if _gate_ok("forecast_data"):
                subtask_futures["forecast"] = thread_pool.default.submit(_forecast)
            else:
                logger.info("[门控] 盈利预测已是最新, 跳过")

            def _income():
                if not (getattr(config, "COLLECT_INCOME_ENABLED", True) and self._should_collect("income_data", 7)):
                    return 0
                try:
                    from scripts.tushare_collect import collect_income
                    collect_income()
                    logger.info("利润表采集完成")
                    return 1
                except Exception as e:
                    logger.warning(f"利润表采集失败: {e}")
                    return 0
            if _gate_ok("income_data"):
                subtask_futures["income"] = thread_pool.default.submit(_income)
            else:
                logger.info("[门控] 利润表已是最新, 跳过")

            def _analyst_reports():
                enabled = getattr(config, "COLLECT_AKSHARE_CONSENSUS_ENABLED", False)
                if not (enabled and self._should_collect("analyst_forecast_report_captures", 1, date_col="captured_at")):
                    return 0
                try:
                    from scripts.tushare_collect import collect_analyst_forecast
                    n = collect_analyst_forecast()
                    logger.info(f"卖方研报预测采集: {n}")
                    return n
                except Exception as e:
                    logger.warning(f"卖方研报预测采集失败: {e}")
                    return 0
            if _gate_ok("analyst_forecast_report_captures"):
                subtask_futures["analyst_reports"] = thread_pool.default.submit(_analyst_reports)
            else:
                logger.info("[门控] 分析师预期已是最新, 跳过")

            def _daily_basic():
                if not getattr(config, "COLLECT_DAILY_BASIC_ENABLED", True):
                    return 0
                try:
                    from core.data.sources.tushare import get_pro
                    from core.data import Database
                    import math

                    def _safe_float(v):
                        if v is None:
                            return None
                        try:
                            f = float(v)
                            if math.isnan(f) or math.isinf(f):
                                return None
                            return f
                        except (ValueError, TypeError):
                            return None

                    pro2 = get_pro()
                    if not pro2:
                        logger.info("[市值] Tushare 不可用")
                        return 0
                    all_dates = [r["trade_date"] for r in Database.fetchall(
                        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date DESC LIMIT 5"
                    )]
                    if all_dates:
                        min_date = all_dates[-1]
                        done = {r["trade_date"] for r in Database.fetchall(
                            "SELECT DISTINCT trade_date FROM daily_basic WHERE pe IS NOT NULL "
                            "AND float_share IS NOT NULL AND trade_date >= %s",
                            (min_date,)
                        )}
                    else:
                        done = set()
                    need = [d for d in all_dates if d not in done]
                    db_saved = 0
                    for d in need:
                        ymd = d.strftime("%Y%m%d")
                        try:
                            df = pro2.daily_basic(trade_date=ymd,
                                fields="ts_code,total_mv,circ_mv,float_share,pe,pb,turnover_rate")
                            if df is None or df.empty:
                                continue
                            values = []
                            for row in df.itertuples(index=False):
                                code = str(row.ts_code).split(".")[0]
                                values.append((code, d.isoformat(),
                                    getattr(row, 'total_mv', None), getattr(row, 'circ_mv', None),
                                    _safe_float(getattr(row, 'float_share', None)),
                                    _safe_float(getattr(row, 'pe', None)),
                                    _safe_float(getattr(row, 'pb', None)),
                                    _safe_float(getattr(row, 'turnover_rate', None))))
                            Database.executemany(
                                "INSERT INTO daily_basic (code, trade_date, total_mv, circ_mv, float_share, pe, pb, turnover_rate) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                                "total_mv=VALUES(total_mv), circ_mv=VALUES(circ_mv), float_share=VALUES(float_share), "
                                "pe=VALUES(pe), pb=VALUES(pb), turnover_rate=VALUES(turnover_rate)",
                                values)
                            db_saved += len(values)
                            logger.info(f"[市值] {ymd}: {len(values)} 只")
                        except Exception as e:
                            logger.warning(f"[市值] {ymd} 失败: {e}")
                    if db_saved:
                        logger.info(f"每日市值采集: {db_saved} 条")
                    return db_saved
                except Exception as e:
                    logger.warning(f"每日市值采集失败: {e}")
                    return 0
            if _gate_ok("daily_basic"):
                subtask_futures["daily_basic"] = thread_pool.default.submit(_daily_basic)
            else:
                logger.info("[门控] 每日市值已是最新, 跳过")

            def _macro():
                if not getattr(config, "MACRO_COLLECT_ENABLED", True):
                    return 0
                try:
                    from scripts.collect_macro import collect_shibor, collect_lpr, collect_money_supply, collect_cpi, collect_ppi, collect_pmi
                    mc = (collect_shibor() or 0)
                    if self._should_collect("macro_lpr", 35, date_col="trade_date"):
                        mc += collect_lpr() or 0
                    if self._should_collect("macro_money_supply", 30, date_col="trade_date"):
                        mc += collect_money_supply() or 0
                    if self._should_collect("macro_cpi", 30, date_col="trade_date"):
                        mc += collect_cpi() or 0
                    if self._should_collect("macro_ppi", 30, date_col="trade_date"):
                        mc += collect_ppi() or 0
                    if self._should_collect("macro_pmi", 30, date_col="trade_date"):
                        mc += collect_pmi() or 0
                    logger.info(f"宏观数据采集: {mc} 条")
                    return mc
                except Exception as e:
                    logger.warning(f"宏观数据采集失败: {e}")
                    return 0
            if any(_gate_ok(t) for t in ["macro_shibor", "macro_lpr", "macro_money_supply",
                                              "macro_cpi", "macro_ppi", "macro_pmi"]):
                subtask_futures["macro"] = thread_pool.default.submit(_macro)
            else:
                logger.info("[门控] 宏观数据全部最新, 跳过")

            def _etf_flow():
                if not getattr(config, "COLLECT_ETF_FLOW_ENABLED", True):
                    return 0
                try:
                    from core.data.sources.etf_flow import collect_etf_fund_flow
                    ef = collect_etf_fund_flow(trade_date=run_date)
                    if ef:
                        logger.info(f"ETF资金流采集: {ef} 条")
                    return ef or 0
                except Exception as e:
                    logger.warning(f"ETF资金流采集失败: {e}")
                    return 0
            if _gate_ok("etf_fund_flow"):
                subtask_futures["etf_flow"] = thread_pool.default.submit(_etf_flow)
            else:
                logger.info("[门控] ETF资金流已是最新, 跳过")

            # ── 收集并行子任务结果 ──
            n = 0; nf = 0; c = 0; f = 0; fc = 0; inc = 0; ac = 0; db_saved = 0; mc = 0; ef = 0
            for key, fut in subtask_futures.items():
                try:
                    val = fut.result(timeout=600)
                except Exception as e:
                    logger.warning(f"[{key}] 子任务异常: {e}")
                    val = 0
                if key == "north_summary": n = val or 0
                elif key == "north_detail": nf = val or 0; result["north_flow"] = nf
                elif key == "capital_flow": c = val or 0; result["capital_flow"] = c
                elif key == "fundamental":
                    if isinstance(val, dict):
                        f = val.get("saved", 0); result["fundamental"] = val
                    else: f = val or 0
                elif key == "forecast": fc = val or 0
                elif key == "income": inc = val or 0
                elif key == "akshare_consensus": ac = val or 0
                elif key == "daily_basic": db_saved = val or 0
                elif key == "macro": mc = val or 0; result["macro"] = mc
                elif key == "etf_flow": ef = val or 0; result["etf_flow"] = ef

            result["forecast"] = fc
            result["income"] = inc
            result["akshare_consensus"] = ac
            result["daily_basic"] = db_saved
            result["etf_flow"] = ef

            # ── 数据完整性校验 + 自动回补 ──
            integrity_ok = True
            try:
                from core.data.integrity import check_and_repair
                integrity = check_and_repair(trade_date=run_date, auto_repair=True)
                result["integrity"] = integrity
                integrity_ok = bool(integrity.get("all_pass", False))
                if integrity.get("all_pass"):
                    logger.info(f"[完整性] {run_date} 全部通过")
                else:
                    repaired = integrity.get("repaired", [])
                    if repaired:
                        logger.info(f"[完整性] 自动回补 {len(repaired)} 项: "
                                    f"{', '.join(r['desc'] for r in repaired)}")
            except Exception as e:
                logger.warning(f"完整性校验失败: {e}")
                result["integrity_error"] = str(e)
                integrity_ok = False

            # ── 阶段E: 补算联动链 ──

            # E1: forward_returns 预计算 (K线更新后自动触发)
            if saved > 0:
                try:
                    from core.domain.factors.forward_returns import get_forward_cache
                    from core.shared.trading_time import get_latest_trading_day
                    fc = get_forward_cache()
                    latest_td = get_latest_trading_day()
                    start = latest_td - datetime.timedelta(days=60)
                    fc.warm_up(start, latest_td, periods=(5, 10, 20, 40))
                    fc.clear_memory()
                    logger.info(f"[补算] forward_returns 已更新 ({start} ~ {latest_td})")
                    result["forward_ret"] = True
                except Exception as e:
                    logger.warning(f"[补算] forward_returns 失败: {e}")

            # E2: 基本面因子缓存重建
            try:
                from core.data.fundamental import preload_cache
                if preload_cache(force=True):
                    logger.info("基本面因子缓存已重建")
            except Exception as e:
                logger.warning(f"基本面因子缓存重建失败: {e}")

            # ── 每日市场状态更新 ──
            try:
                from core.domain.regime.history import ensure_today
                ensure_today()
                logger.info("市场状态已更新")
            except Exception as e:
                logger.warning(f"市场状态更新失败: {e}")

            # ── 等待后台 Qlib 同步完成 ──
            if qlib_future is not None:
                try:
                    sync_result = qlib_future.result(timeout=600)
                    if sync_result:
                        result["qlib_sync"] = sync_result
                        result["new_dates"] = sync_result.get("new_dates", 0)
                        result["new_stocks"] = sync_result.get("new_stocks", 0)
                        if sync_result.get("new_dates", 0) > 0:
                            def _prewarm_lgb_lstm():
                                try:
                                    from core.data import SettingsRepo
                                    warm = SettingsRepo.get("lgb_lstm_prewarm", {}) or {}
                                    models = warm.get("models") or {}
                                    pools = warm.get("pools") or []
                                    if not models or not pools:
                                        return
                                    from core.adapters.qlib.models import QlibModelManager
                                    from core.shared.disabled_models import is_disabled
                                    from core.shared import config as app_config
                                    from qlib.data import D
                                    from pathlib import Path
                                    calendar = D.calendar()
                                    if not len(calendar):
                                        return
                                    predict_date = str(calendar[-1])[:10]
                                    manager = QlibModelManager()
                                    seen = set()
                                    for model in models.values():
                                        if not isinstance(model, (list, tuple)) or len(model) != 2:
                                            continue
                                        exp_id, run_id = str(model[0]), str(model[1])
                                        if (exp_id, run_id) in seen or is_disabled(exp_id, run_id):
                                            continue
                                        artifacts = (Path(str(app_config.QLIB_MLFLOW_URI).replace("file:", "")) /
                                                     exp_id / run_id / "artifacts")
                                        if not (artifacts / "params_lstm.pkl").exists():
                                            continue
                                        seen.add((exp_id, run_id))
                                        for instruments in pools:
                                            manager.predict_dual(
                                                exp_id, run_id,
                                                predict_start=predict_date,
                                                predict_end=predict_date,
                                                instruments=instruments,
                                            )
                                            logger.info(
                                                f"[LGB+LSTM] 每日预测预热完成: {exp_id}/{run_id} "
                                                f"pool={instruments} date={predict_date}")
                                except Exception as e:
                                    logger.warning(f"[LGB+LSTM] 每日预测预热失败: {e}")
                            if self._qlib_read_fn:
                                thread_pool.default.submit(self._qlib_read_fn, _prewarm_lgb_lstm)
                            else:
                                logger.error("Qlib预测预热未运行：调度器未配置generation读租约")
                except Exception as e:
                    logger.warning(f"Qlib 同步后台任务异常: {e}")
                    result["qlib_error"] = str(e)

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            result["elapsed_seconds"] = elapsed
            result["status"] = "success" if integrity_ok and result.get("kline_complete", True) else "partial"
            self.last_result = result
            logger.info(f"每日采集完成: K线{saved}只, 市值{db_saved}条, 北向{n}只, 资金流向{c}只, 基本面{f}条, 一致预期{fc}, 利润表{inc}, akshare{ac}条, 宏观{mc}条, {elapsed:.0f}s")

        except Exception as e:
            logger.exception(f"每日采集失败: {e}")
            result["status"] = "error"
            result["error_message"] = str(e)
            self.last_result = result
            self._notify_failure(run_date, str(e))

        finally:
            self.last_collection = datetime.datetime.now().isoformat()
            self.collection_count += 1
            self._log_collection(result)
            self._busy = False

    def _log_collection(self, result):
        try:
            from core.data import Database
            Database.execute(
                "INSERT INTO daily_collection_log "
                "(source, data_source, stock_pool, total_codes, saved, new_dates, "
                "new_stocks, elapsed_seconds, status, error_message) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    result.get("source", "unknown"),
                    result.get("data_source", self.data_source),
                    result.get("stock_pool", self.stock_pool),
                    result.get("total_codes", 0),
                    result.get("saved", 0),
                    result.get("new_dates", 0),
                    result.get("new_stocks", 0),
                    result.get("elapsed_seconds", 0),
                    result.get("status", "error"),
                    result.get("error_message", ""),
                ),
            )
        except Exception as e:
            logger.warning(f"采集日志写入失败: {e}")

    def _should_collect(self, table, min_days, date_col="created_at"):
        """检查上次采集是否在 min_days 天内，超过则需重新采集。当日结果缓存避免重复 DB 查询。"""
        today = datetime.date.today().isoformat()
        cache_key = f"{table}:{date_col}"
        if cache_key in self._collect_day_cache:
            cached_day, cached_val = self._collect_day_cache[cache_key]
            if cached_day == today:
                return cached_val
        try:
            from core.data import Database
            row = Database.fetchone(f"SELECT MAX({date_col}) as last_dt FROM {table}")
            if row and row["last_dt"]:
                last_dt = row["last_dt"]
                if hasattr(last_dt, "date"):
                    last_dt = last_dt.date()
                elif hasattr(last_dt, "strftime"):
                    last_dt = datetime.date.fromisoformat(str(last_dt)[:10])
                delta = (datetime.date.today() - last_dt).days
                if delta < min_days:
                    logger.info(f"[{table}] {delta}天前已采集 (<{min_days}天), 跳过")
                    self._collect_day_cache[cache_key] = (today, False)
                    return False
        except Exception:
            pass
        self._collect_day_cache[cache_key] = (today, True)
        return True

    def _notify_failure(self, run_date, error):
        if not getattr(config, "COLLECT_NOTIFY_ON_FAILURE", True):
            return
        try:
            from core.shared.infrastructure.notify import get_wecom
            n = get_wecom()
            if n.enabled:
                n.send_markdown(
                    f"## A股数据采集失败\n"
                    f"**日期**: {run_date}\n"
                    f"**数据源**: {self.data_source}\n"
                    f"**错误**: {error}\n"
                    f"**时间**: {datetime.datetime.now().strftime('%H:%M:%S')}"
                )
        except Exception:
            pass

    def _collect_concept_sectors(self):
        """批量采集全部概念板块成分股 → MySQL stock_sectors (sector_type='concept')
        共享 cookie/代码映射，并发爬取，每周执行一次。
        """
        import requests
        import re
        from bs4 import BeautifulSoup
        from concurrent.futures import as_completed

        logger.info("[概念采集] 开始批量采集概念板块成分股...")
        t0 = time.time()
        result = {"succeeded": 0, "failed": 0, "failed_concepts": [], "source_overrides": {}, "total_inserted": 0}

        try:
            from core.data.sources.akshare import call_ak
            concept_df = call_ak("stock_board_concept_name_ths")
            name_col = "name" if "name" in concept_df.columns else concept_df.columns[0]
            code_col = "code" if "code" in concept_df.columns else concept_df.columns[1]
            concepts = [(str(getattr(r, name_col)), str(getattr(r, code_col))) for r in concept_df.itertuples(index=False)]
            result["concept_count"] = len(concepts)
            from core.data import Database
            prior = Database.fetchone(
                "SELECT COUNT(DISTINCT sector_name) AS n FROM stock_sectors WHERE sector_type='concept'"
            ) or {}
            prior_concept_count = int(prior.get("n") or 0)
            minimum_concepts = max(100, int(prior_concept_count * 0.5)) if prior_concept_count else 100
            if len(concepts) < minimum_concepts:
                result["status"] = "error"
                result["expected_concepts_min"] = minimum_concepts
                result["source_concepts"] = len(concepts)
                result["error_message"] = "概念板块源覆盖显著低于现有快照，拒绝发布"
                logger.error(f"[概念采集] 源仅返回 {len(concepts)} 个板块，现有 {prior_concept_count} 个，低于发布下限 {minimum_concepts}")
                return result
            logger.info(f"[概念采集] 共 {len(concepts)} 个概念板块")
            fallbacks = {}
            if any(name == "融资融券" for name, _ in concepts):
                from core.data.sources.tushare import get_pro
                from core.shared.trading_time import get_latest_trading_day
                pro = get_pro()
                if not pro:
                    raise RuntimeError("融资融券官方成分回退源不可用")
                source_date = get_latest_trading_day()
                margin_df = pro.margin_secs(trade_date=source_date.strftime("%Y%m%d"))
                if margin_df is None or margin_df.empty or "ts_code" not in margin_df.columns:
                    raise RuntimeError("融资融券官方成分返回空数据")
                active_codes = {
                    str(row["symbol"]).zfill(6)
                    for row in Database.fetchall(
                        "SELECT symbol FROM stock_identity WHERE list_status='L' AND symbol IS NOT NULL"
                    )
                }
                margin_codes = sorted({
                    str(ts_code).split(".")[0]
                    for ts_code in margin_df["ts_code"]
                    if str(ts_code).split(".")[0] in active_codes
                })
                if len(margin_codes) < 100:
                    raise RuntimeError(f"融资融券官方成分覆盖异常: {len(margin_codes)} 只")
                fallbacks["融资融券"] = margin_codes
                result["source_overrides"]["融资融券"] = {
                    "source": "TuShare margin_secs",
                    "trade_date": source_date.isoformat(),
                    "stock_count": len(margin_codes),
                }
            time.sleep(60)
        except Exception as e:
            logger.error(f"[概念采集] 获取概念板块列表失败: {e}")
            return {"status": "error", "error_message": str(e)}

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }
        from core.data.sources.akshare import get_mini_racer
        from akshare.datasets import get_ths_js
        from core.shared.persistence import load_text

        js_code = get_mini_racer()
        js_code.eval(load_text(get_ths_js("ths.js")))
        headers["Cookie"] = f"v={js_code.call('v')}"

        concurrency = getattr(config, "COLLECT_CONCEPT_SECTORS_CONCURRENCY", 6)
        sem = threading.Semaphore(concurrency)
        lock = threading.Lock()
        request_lock = threading.Lock()
        last_request_at = 0.0
        all_rows = []
        total = len(concepts)
        done = 0

        def _scrape_one(name, code):
            nonlocal done
            with sem:
                if name in fallbacks:
                    with lock:
                        all_rows.extend((c, name, "concept", "tushare.margin_secs", code) for c in fallbacks[name])
                        done += 1
                        if done % 50 == 0 or done == total:
                            logger.info(f"[概念采集] 进度: {done}/{total}")
                    return True
                succeeded = False
                last_error = None
                session = requests.Session()
                session.headers.update(headers)
                referer_url = "https://q.10jqka.com.cn/"
                for board_attempt in range(2):
                    codes = []
                    try:
                        def _get(url):
                            nonlocal last_request_at, referer_url
                            request_error = None
                            for attempt in range(3):
                                try:
                                    with request_lock:
                                        wait_seconds = 3.0 - (time.monotonic() - last_request_at)
                                        if wait_seconds > 0:
                                            time.sleep(wait_seconds)
                                        session.headers["Referer"] = referer_url
                                        response = session.get(url, timeout=20, allow_redirects=True)
                                        last_request_at = time.monotonic()
                                    if response.status_code != 200:
                                        raise RuntimeError(f"HTTP {response.status_code}: {url} -> {response.url}")
                                    if "/account/login" in response.url:
                                        raise RuntimeError(f"分页被重定向至登录页: {url} -> {response.url}")
                                    referer_url = response.url
                                    return response
                                except Exception as e:
                                    request_error = e
                                    if attempt < 2:
                                        time.sleep(1.0 + attempt)
                            raise request_error

                        url = f"https://q.10jqka.com.cn/gn/detail/code/{code}/"
                        r = _get(url)
                        soup = BeautifulSoup(r.text, "lxml")
                        page_span = soup.find(name="span", attrs={"class": "page_info"})
                        page_match = re.search(r"/\s*(\d+)", page_span.get_text(" ", strip=True)) if page_span else None
                        total_pages = int(page_match.group(1)) if page_match else 1
                        if total_pages > 1000:
                            raise RuntimeError(f"分页数异常: {total_pages}")
                        for page in range(1, total_pages + 1):
                            if page == 1:
                                ps = soup
                            else:
                                pu = f"https://q.10jqka.com.cn/gn/detail/code/{code}/order/desc/page/{page}/"
                                pr = _get(pu)
                                ps = BeautifulSoup(pr.text, "lxml")
                            page_codes = []
                            for table in ps.find_all("table"):
                                for row in table.find_all("tr")[1:]:
                                    cols = row.find_all("td")
                                    if len(cols) >= 2:
                                        ct = cols[1].get_text(strip=True)
                                        if ct.isdigit() and len(ct) == 6:
                                            page_codes.append(ct)
                            if not page_codes:
                                raise RuntimeError(f"第 {page} 页未解析到成分股")
                            codes.extend(page_codes)
                        if not codes:
                            raise RuntimeError("未解析到有效板块成分")
                        with lock:
                            all_rows.extend((c, name, "concept", "ths.10jqka_html", code) for c in codes)
                        succeeded = True
                        break
                    except Exception as e:
                        last_error = e
                        if board_attempt == 0:
                            time.sleep(2)
                if not succeeded:
                    logger.warning(f"[概念采集] 板块失败 {name}({code}): {last_error}")
                with lock:
                    if not succeeded:
                        result["failed_concepts"].append({"name": name, "code": code, "error": str(last_error)})
                    done += 1
                    if done % 50 == 0 or done == total:
                        logger.info(f"[概念采集] 进度: {done}/{total}")
                return succeeded

        futures = [thread_pool.default.submit(_scrape_one, name, code) for name, code in concepts]
        for f in as_completed(futures):
            if f.result():
                result["succeeded"] += 1
            else:
                result["failed"] += 1

        if result["failed"]:
            result["status"] = "partial"
            result["error_message"] = "存在采集失败板块，未发布不完整的概念成员快照"
            result["elapsed"] = round(time.time() - t0, 1)
            logger.warning(f"[概念采集] {result['failed']} 个板块失败，保留现有概念成员关系不变")
            return result

        if all_rows:
            try:
                from core.data import Database
                from core.data.stock_sectors_migration import ensure_stock_sector_observations

                all_rows = list(dict.fromkeys(all_rows))
                observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                observation_rows = [
                    (observed_at, source, sector_code, sector_name, code)
                    for code, sector_name, _, source, sector_code in all_rows
                ]
                current_rows = list({
                    (code, sector_name, sector_type, observed_at)
                    for code, sector_name, sector_type, _, _ in all_rows
                })
                ensure_stock_sector_observations()

                def _replace_concepts(cursor):
                    cursor.execute("DELETE FROM stock_sectors WHERE sector_type='concept'")
                    for i in range(0, len(current_rows), 400):
                        chunk = current_rows[i:i + 400]
                        cursor.executemany(
                            "INSERT INTO stock_sectors (code, sector_name, sector_type, updated_at) "
                            "VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at)",
                            chunk,
                        )
                    for i in range(0, len(observation_rows), 1000):
                        cursor.executemany(
                            "INSERT IGNORE INTO stock_sector_membership_observations "
                            "(observed_at,source,sector_code,sector_name,code) VALUES (%s,%s,%s,%s,%s)",
                            observation_rows[i:i + 1000],
                        )
                Database.transaction(_replace_concepts)
                result["total_inserted"] = len(current_rows)
                result["observed_relations"] = len(observation_rows)
                result["observed_at"] = observed_at.isoformat()
                result["source_counts"] = {
                    source: sum(1 for row in observation_rows if row[1] == source)
                    for source in {row[1] for row in observation_rows}
                }
                logger.info(f"[概念采集] 写入 stock_sectors: {len(current_rows)} 条 (概念板块)")
            except Exception as e:
                logger.error(f"[概念采集] 写入MySQL失败: {e}")
                result["status"] = "error"
                result["error_message"] = str(e)
                return result
        elif result["succeeded"]:
            result["status"] = "error"
            result["error_message"] = "采集成功但未解析到任何成分股"
            return result

        elapsed = time.time() - t0
        result["status"] = "success" if result["failed"] == 0 else "partial"
        result["elapsed"] = round(elapsed, 1)
        logger.info(f"[概念采集] 完成: {result['succeeded']}成功 {result['failed']}失败, "
                    f"{result['total_inserted']}条, {elapsed:.0f}s")
        return result


_collect_scheduler = None


def get_collect_scheduler() -> DataCollectionScheduler:
    global _collect_scheduler
    if _collect_scheduler is None:
        _collect_scheduler = DataCollectionScheduler()
    return _collect_scheduler


class SmokeTestScheduler:

    def __init__(self):
        self._registered = False
        self._busy = False
        self.smoke_time = getattr(config, "SMOKE_TEST_TIME", "04:00")
        self._smoke_t = _parse_time(self.smoke_time)
        self.last_run = None
        self.last_result = None
        self.count = 0

    @property
    def running(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            meta = get_unified_scheduler().get_meta("smoke-test")
            return meta is not None and meta.get("scheduled", False)
        except Exception:
            return False

    @property
    def status(self):
        try:
            from core.shared.infrastructure.scheduler import get_unified_scheduler
            meta = get_unified_scheduler().get_meta("smoke-test") or {}
        except Exception:
            meta = {}

        return {
            "running": self.running,
            "busy": self._busy or meta.get("running", False),
            "smoke_time": self.smoke_time,
            "last_run": self.last_run,
            "last_result": self.last_result,
            "next_run": meta.get("next_run"),
            "count": self.count,
        }

    def start(self):
        if not getattr(config, "SMOKE_TEST_ENABLED", True):
            logger.info("每日冒烟测试未启用 (SMOKE_TEST_ENABLED=false)")
            return
        if self._registered:
            return

        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()
        cron = f"{self._smoke_t.minute} {self._smoke_t.hour} * * mon-fri"

        def _do():
            self._do_smoke(source="scheduled")

        meta = mgr.get_meta("smoke-test")
        if meta is None:
            mgr.register("smoke-test", _do, cron, trading_day_only=False)
        elif not meta.get("scheduled"):
            mgr.resume("smoke-test")

        if not mgr._started:
            mgr.start()

        self._registered = True
        logger.info(f"每日冒烟测试已启动 (工作日 {self.smoke_time})")

    def stop(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        try:
            get_unified_scheduler().pause("smoke-test")
        except Exception:
            pass
        self._registered = False
        logger.info("每日冒烟测试已停止")

    def trigger(self):
        thread_pool.light.submit(self._do_smoke, source=SESSION_MANUAL)

    def _do_smoke(self, source="scheduled"):
        if self._busy:
            logger.warning("上一次冒烟测试尚未完成，跳过本次")
            return
        self._busy = True
        try:
            from core.shared.infrastructure.smoke_test import SmokeTestRunner
            report = SmokeTestRunner().run_all(source=source)
            self.last_result = f"{report['passed']}/{report['total']} 通过" + (
                f", {report['failed']} 失败" if report["failed"] else "")
            logger.info(f"冒烟测试完成: {self.last_result}, {report['elapsed_seconds']}s")
        except Exception as e:
            logger.exception(f"冒烟测试执行失败: {e}")
            self.last_result = f"执行异常: {e}"
        finally:
            self.last_run = datetime.datetime.now().isoformat()
            self.count += 1
            self._busy = False


_smoke_scheduler = None


def get_smoke_scheduler() -> SmokeTestScheduler:
    global _smoke_scheduler
    if _smoke_scheduler is None:
        _smoke_scheduler = SmokeTestScheduler()
    return _smoke_scheduler


# ── 同花顺数据采集调度器 ──────────────────────────────────────

class THSCollectorScheduler:
    """
    同花顺 Web API 数据采集调度器

    三个任务:
      ths-realtime   — 盘中每12s采集实时行情 (80字段→Redis+MySQL)
      ths-intraday   — 盘中每分钟采集分时数据
      ths-eod        — 收盘15:05采集日线和60分钟K线
    """

    def __init__(self):
        self._registered = False
        self._busy = False

    def start(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        from core.data.ths_collector import ensure_tables

        ensure_tables()
        mgr = get_unified_scheduler()

        # 只有EOD定时任务: 收盘后增量更新日K+60分钟K线
        if mgr.get_meta("ths-eod") is None:
            def _do_eod():
                from core.data.ths_collector import THSCollector, _get_cookie, collect_cycle
                import time
                try:
                    codes = _get_ths_codes()
                    c = THSCollector()
                    c.session.headers["Cookie"] = f"v={_get_cookie()}"
                    collect_cycle(codes, c, "daily")
                    time.sleep(30)
                    collect_cycle(codes, c, "60min")
                    logger.info(f"[ths-eod] 收盘采集完成: {len(codes)}只")
                except Exception as e:
                    logger.warning(f"[ths-eod] 采集异常: {e}")

            mgr.register("ths-eod", _do_eod, "5 15 * * mon-fri", trading_day_only=True)

        if not mgr._started:
            mgr.start()

        self._registered = True
        logger.info("[ths-collector] 同花顺EOD采集已启动 (仅收盘15:05增量, 实时数据按需拉取)")

        try:
            from core.data.collection_gate import GateCheckService
            now = datetime.datetime.now()
            gate = GateCheckService().check_table("ths_daily_kline", force=True)
            if is_trading_day(now.date()) and (now.hour, now.minute) >= (15, 5) and gate.should_collect:
                logger.warning(f"同花顺日K收盘后仍未完成，启动补跑: {gate.reason}")
                mgr.trigger("ths-eod")
        except Exception as e:
            logger.warning(f"同花顺启动补跑检查失败: {e}")

    def stop(self):
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        mgr = get_unified_scheduler()
        for name in ("ths-eod",):
            try:
                mgr.pause(name)
            except Exception:
                pass
        self._registered = False
        logger.info("[ths-collector] 同花顺EOD采集已停止")


def _get_ths_codes():
    """获取同花顺监控标的列表 (上限150只, 防反爬)"""
    import random
    max_codes = getattr(config, "THS_MAX_CODES", 150)

    try:
        from core.data import Database
        from core.data.repos.local_ledger import ACCOUNT_ID
        rows = Database.fetchall(
            "SELECT DISTINCT code FROM holdings WHERE account_id=%s AND shares > 0 LIMIT %s",
            (ACCOUNT_ID, max_codes)
        )
        if rows:
            codes = sorted(r["code"] for r in rows)
            random.shuffle(codes)
            return codes[:max_codes]
    except Exception as e:
        logger.debug(f"[ths] 从持仓取代码失败: {e}")

    try:
        from core.domain.factors.pool import FactorPoolManager
        mgr = FactorPoolManager()
        factors = mgr.get_active_factors()
        codes = set()
        for f in factors:
            code = f.get("code") or f.get("stock_code")
            if code and str(code).isdigit() and len(str(code)) == 6:
                codes.add(str(code))
        if codes:
            result = sorted(codes)[:max_codes]
            random.shuffle(result)
            return result
    except Exception as e:
        logger.debug(f"[ths] 从因子池取代码失败: {e}")

    return getattr(config, "THS_MONITOR_CODES", ["600519", "000001", "300750"])


_ths_collector_scheduler = None


def get_ths_collector_scheduler() -> THSCollectorScheduler:
    global _ths_collector_scheduler
    if _ths_collector_scheduler is None:
        _ths_collector_scheduler = THSCollectorScheduler()
    return _ths_collector_scheduler
