import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══ 必须在任何其他库之前 import akshare ═══
# py_mini_racer (V8) 的 partition_address_space 只能初始化一次。
# 如果 numpy/scipy/torch 先配置了内存分配器，V8 初始化就会 FATAL 崩溃。
# 必须在进程启动的最早期、单线程环境下完成 V8 初始化。
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ═══ 必须在任何其他库之前初始化 V8 引擎 ═══
import py_mini_racer
py_mini_racer.MiniRacer()

def _warmup_akshare():
    import akshare as _ak
    import logging
    _log = logging.getLogger("akshare_warmup")
    try:
        for _fn in (_ak.stock_zh_a_spot_em, _ak.tool_trade_date_hist_sina, _ak.stock_info_a_code_name):
            try: _fn()
            except Exception: pass
        _log.info("akshare 预热完成")
    except Exception as e:
        _log.warning(f"akshare 预热失败: {e}")

import logging
from core.shared.infrastructure.thread_pool import thread_pool
from core.shared import config
import math
import json
import asyncio
import threading
from typing import Any

from core.shared.logging import setup_logging, get_logger
setup_logging()

logger = get_logger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager


def _sanitize_float(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _sanitize_float(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_float(item) for item in v]
    return v


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        sanitized = _sanitize_float(content)
        return json.dumps(
            sanitized,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# ═══ 启动就绪事件 (替代 sleep 链) ═══
_db_ready = asyncio.Event()
_qlib_ready = threading.Event()

# ═══ 组件健康注册表 ═══
# importance: "required"→挂了fatal, "preferred"→degraded, "optional"→仅提示
_components = {
    "db":          {"status": "pending", "importance": "required",  "detail": ""},
    "redis":       {"status": "pending", "importance": "preferred", "detail": ""},
    "factors":     {"status": "pending", "importance": "required",  "detail": ""},
    "baostock":    {"status": "pending", "importance": "preferred", "detail": ""},
    "schedulers":  {"status": "pending", "importance": "required",  "detail": ""},
    "qlib":        {"status": "pending", "importance": "optional",  "detail": ""},
    "akshare":     {"status": "pending", "importance": "optional",  "detail": ""},
}

_schedulers_started = False
_wired = False
_components_lock = threading.Lock()


def _update_component(name, status, detail=""):
    with _components_lock:
        _components[name]["status"] = status
        _components[name]["detail"] = detail


_SCHEDULERS = (
    ("risk",        lambda: (__import__("core.shared.infrastructure.jobs", fromlist=["get_scheduler"]).get_scheduler(), "start", "stop")),
    ("collect",     lambda: (__import__("core.shared.infrastructure.jobs", fromlist=["get_collect_scheduler"]).get_collect_scheduler(), "start", "stop")),
    ("smoke",       lambda: (__import__("core.shared.infrastructure.jobs", fromlist=["get_smoke_scheduler"]).get_smoke_scheduler(), "start", "stop")),
    ("ths",         lambda: (__import__("core.shared.infrastructure.jobs", fromlist=["get_ths_collector_scheduler"]).get_ths_collector_scheduler(), "start", "stop")),
    ("sentiment",   lambda: (__import__("core.adapters.nlp.scheduler", fromlist=["get_sentiment_scheduler"]).get_sentiment_scheduler(), "start", "stop")),
    ("notice",      lambda: (__import__("core.adapters.nlp.notice_scheduler", fromlist=["get_notice_scheduler"]).get_notice_scheduler(), "start", "stop")),
    ("unified",     lambda: (__import__("core.shared.infrastructure.scheduler", fromlist=["get_unified_scheduler"]).get_unified_scheduler(), "start", "stop")),
)


def _stop_schedulers():
    global _schedulers_started
    if not _schedulers_started:
        return
    try:
        for _name, _getter in _SCHEDULERS:
            _inst, _start_method, _stop_method = _getter()
            getattr(_inst, _stop_method)()
        _schedulers_started = False
        _update_component("schedulers", "ok", "")
        logger.info("[Shutdown] 调度器已停止")
    except Exception as e:
        logger.warning(f"[Shutdown] 调度器停止异常: {e}")


def _wire_scheduler_dependencies():
    """将 domain 层的具体任务实现注入 shared/infrastructure 的调度器框架。

    解决 shared → domain 的分层违规：调度器只负责 WHEN，WHAT 由上层注入。
    分类捕获异常，单类注入失败不阻止其他类，但用 ERROR 级别确保监控可感知。
    使用 _wired 标志允许健康检查重试调用时补注入。
    """
    global _wired
    if _wired:
        return
    errors = []

    try:
        from core.shared.infrastructure.jobs import get_scheduler, get_collect_scheduler
    except Exception as e:
        logger.error(f"[Wiring] 无法导入调度器模块: {e}")
        raise

    # -- 止盈止损监控告警 --
    try:
        from core.domain.portfolio.risk_alert import run_stop_alert_scan
        get_scheduler(scan_fn=run_stop_alert_scan)
        from backend.service.risk_notification_outbox import dispatch_risk_notification_outbox
        from core.shared.infrastructure.scheduler import get_unified_scheduler
        unified_scheduler = get_unified_scheduler()
        if unified_scheduler.get_meta("risk-notification-outbox") is None:
            unified_scheduler.register_interval(
                "risk-notification-outbox",
                dispatch_risk_notification_outbox,
                30,
                max_instances=1,
            )
    except Exception as e:
        errors.append(f"止盈止损告警: {e}")
        logger.error(f"[Wiring] 止盈止损告警注入失败: {e}")

    # -- 数据采集子任务 --
    try:
        from core.domain.screening._scoring_legacy import fetch_and_save_north_flow, fetch_and_save_capital_flow
        from backend.service.fundamental_collection import fetch_and_save_fundamentals
        from core.data.integrity import set_fundamental_collection
        set_fundamental_collection(fetch_and_save_fundamentals)
        cs = get_collect_scheduler()
        from backend.service.qlib_access import run_with_qlib_read_access
        cs.set_qlib_read_fn(run_with_qlib_read_access)
        cs.register_task("north_detail", lambda d: fetch_and_save_north_flow(date=d))
        cs.register_task("capital_flow", lambda d: fetch_and_save_capital_flow(date=d))
        cs.register_task("fundamental", lambda d: fetch_and_save_fundamentals(max_workers=4))
        from scripts.collect_new_sources import (
            collect_margin_incremental,
            collect_lhb_incremental,
            collect_shareholder_incremental,
        )
        cs.register_task("margin_daily", lambda d: collect_margin_incremental(d.strftime("%Y%m%d")))
        cs.register_task("lhb_daily", lambda d: collect_lhb_incremental(d.strftime("%Y%m%d")))
        cs.register_task("shareholder_quarterly", lambda d: collect_shareholder_incremental())
        from backend.service.stock_context import (
            collect_dividend_events,
            collect_stock_concept_membership,
            collect_stock_identity,
            collect_stock_index_membership,
        )
        cs.register_concept_sector_collector(collect_stock_concept_membership)

        def _collect_stock_context(_date):
            result = {}
            for name, collect in (
                ("identity", collect_stock_identity),
                ("industry_membership", collect_stock_index_membership),
                ("dividends", lambda: collect_dividend_events(days=7)),
            ):
                try:
                    result[name] = collect()
                except Exception as e:
                    logger.warning(f"{name} 快照采集失败: {e}")
                    result[name] = None
            return result

        cs.register_task("stock_context", _collect_stock_context)
    except Exception as e:
        errors.append(f"数据采集: {e}")
        logger.error(f"[Wiring] 数据采集子任务注入失败: {e}")

    if errors:
        logger.error(f"[Wiring] {len(errors)} 项注入失败, 系统降级运行: {errors}")
    else:
        _wired = True
        logger.info("[Wiring] 调度器领域任务注入完成")


def _ensure_schedulers():
    global _schedulers_started
    if _schedulers_started:
        return
    try:
        _wire_scheduler_dependencies()

        for _name, _getter in _SCHEDULERS:
            _inst, _start_method, _stop_method = _getter()
            getattr(_inst, _start_method)()

        from core.shared.infrastructure.async_task import task_manager

        def _cleanup_jobs():
            task_manager.cleanup(max_age_seconds=86400)
            try:
                from core.shared.error_store import auto_cleanup
                auto_cleanup()
            except Exception:
                pass
            try:
                from core.adapters.qlib.config import cleanup_dataset_cache
                cleanup_dataset_cache(max_age_days=7)
            except Exception:
                pass

        _inst, _, _ = _SCHEDULERS[-1][1]()
        _inst.register("cleanup-zombie",
            _cleanup_jobs,
            "3 3 * * *", trading_day_only=False)
        _schedulers_started = True
        _update_component("schedulers", "ok", "")
        logger.info("调度器已启动 (含止盈止损告警+采集+情绪+僵尸清理)")
    except Exception as e:
        _update_component("schedulers", "error", str(e))
        logger.warning(f"调度器启动延迟: {e}")
        try:
            from core.shared.error_store import log_error
            import traceback
            log_error(level="error", source="scheduler", exc_type=type(e).__name__,
                      exc_msg=str(e), traceback=traceback.format_exc())
        except Exception:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()

    # ═══ startup: 配置校验 ═══
    def _validate_config():
        try:
            result = config.validate_config()
            if result["errors"]:
                logger.error(f"[Config] {result['errors']} 项配置错误:")
                for lv, msg in result["details"]:
                    if lv == "error":
                        logger.error(f"  ❌ {msg}")
            if result["warnings"]:
                logger.warning(f"[Config] {result['warnings']} 项配置警告:")
                for lv, msg in result["details"]:
                    if lv == "warning":
                        logger.warning(f"  ⚠️  {msg}")
            if not result["errors"] and not result["warnings"]:
                logger.info("[Config] 配置校验通过")
        except Exception as e:
            logger.error(f"[Config] 配置校验失败: {e}", exc_info=True)

    _validate_config()

    # ═══ startup: 并行初始化 ═══
    def _init_database():
        for attempt in range(10):
            try:
                from core.data import Database
                if Database.is_available():
                    Database.create_tables()
                    if not Database.fetchone("SHOW TABLES LIKE 'holdings_adjustments'"):
                        raise RuntimeError("持仓调整审计表未创建")
                    from core.data.daily_quotes_turnover_migration import ensure_daily_quotes_turnover_precision
                    ensure_daily_quotes_turnover_precision()
                    from core.data.daily_basic_float_share_migration import ensure_daily_basic_float_share
                    ensure_daily_basic_float_share()
                    from core.data.fundamental_data_migration import ensure_fundamental_data_columns
                    ensure_fundamental_data_columns()
                    from core.data.income_data_migration import ensure_income_data_oper_cost
                    ensure_income_data_oper_cost()
                    from core.data.income_data_migration import ensure_income_data_version_fields
                    ensure_income_data_version_fields()
                    from core.data.repos.fundamental_versions import ensure_fundamental_versions_table
                    ensure_fundamental_versions_table()
                    from core.data.repos.analyst_forecast_reports import ensure_analyst_forecast_reports_tables
                    ensure_analyst_forecast_reports_tables()
                    from core.data.repos.dividend_source_versions import ensure_dividend_source_versions
                    ensure_dividend_source_versions()
                    from core.data.macro_lpr_migration import ensure_macro_lpr_source_columns
                    ensure_macro_lpr_source_columns()
                    from core.data.north_flow_migration import ensure_north_flow_quarantine
                    ensure_north_flow_quarantine()
                    from core.data.stock_sectors_migration import ensure_stock_sector_type_key
                    ensure_stock_sector_type_key()
                    from core.data.stock_sectors_migration import ensure_stock_sector_observations
                    ensure_stock_sector_observations()
                    from core.data.stock_industry_history_migration import ensure_stock_industry_history_tables
                    ensure_stock_industry_history_tables()
                    from core.data.local_ledger_migration import ensure_local_ledger_schema
                    ensure_local_ledger_schema()
                    _update_component("db", "ok")
                    logger.info(f"数据库表已就绪 (第{attempt+1}次尝试)")
                    break
            except Exception as e:
                logger.warning(f"数据库连接中... (第{attempt+1}次: {e})")
            import time
            time.sleep(1.5)
        else:
            _update_component("db", "error", "10次重试均失败")
            logger.warning("数据库初始化已跳过，系统降级运行")
        loop.call_soon_threadsafe(_db_ready.set)

    def _init_redis():
        try:
            from core.shared.infrastructure.redis_client import ensure_redis_running
            from core.shared import config as _cfg
            if not _cfg.REDIS_ENABLED:
                _update_component("redis", "ok", "已禁用 (REDIS_ENABLED=false)")
                return
            ok = ensure_redis_running()
            _update_component("redis", "ok" if ok else "error",
                              "" if ok else "连接失败, 回退 diskcache")
        except Exception as e:
            _update_component("redis", "error", str(e))
            logger.warning(f"Redis 初始化跳过: {e}")

    def _init_akshare_warmup():
        try:
            _warmup_akshare()
            _update_component("akshare", "ok")
        except Exception as e:
            _update_component("akshare", "error", str(e))

    _startup_futures = []
    _startup_futures.append(("database", thread_pool.light.submit(_init_database)))
    _startup_futures.append(("redis", thread_pool.light.submit(_init_redis)))
    _startup_futures.append(("akshare", thread_pool.light.submit(_init_akshare_warmup)))

    # ═══ startup: Qlib 数据 + 预热 ═══
    def _init_qlib():
        from core.shared import config as cfg
        data_dir = cfg.QLIB_DATA_DIR
        cal_file = data_dir / "calendars" / "day.txt"
        if not cal_file.exists() and cfg.MYSQL_ENABLED:
            try:
                from core.adapters.qlib.bridge import QlibDataBridge
                bridge = QlibDataBridge(data_dir)
                result = bridge.sync_all()
                logger.info(f"qlib 数据初始化完成: {result}")
            except Exception as e:
                logger.warning(f"qlib 数据初始化跳过 (数据库可能无数据): {e}")

        thread_pool.light.submit(_warmup_qlib)

    def _warmup_qlib():
        try:
            from core.adapters.qlib.config import init_qlib
            init_qlib()
            _update_component("qlib", "ok")
            _qlib_ready.set()
            logger.info("[Startup] Qlib 预热完成")
        except Exception as e:
            _update_component("qlib", "error", str(e))
            logger.warning(f"[Startup] Qlib 预热跳过: {e}")

    _startup_futures.append(("qlib_data", thread_pool.light.submit(_init_qlib)))

    # ═══ startup: 延迟初始化 (Event 驱动替代 sleep) ═══
    async def _delayed_init():
        # 等 DB 就绪 (最多 60s)
        try:
            await asyncio.wait_for(_db_ready.wait(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning("[Startup] DB 初始化超时 (60s)，跳过因子/行业初始化")
            return

        def _init_factors():
            all_ok = True
            try:
                from core.domain.factors.registry import get_pool
                pool = get_pool()
                existing = pool.get_all_factors()
                if not any(f.get("active") for f in existing.values() if isinstance(f, dict)):
                    logger.info("[Startup] 因子池为空，正在注册内置因子...")
                    from core.domain.factors.whitebox import FACTOR_TEMPLATES
                    registered = 0
                    for key, tmpl in FACTOR_TEMPLATES.items():
                        if not tmpl.get("category"):
                            continue
                        try:
                            pool.add_factor(key, tmpl.get("label", key), category=tmpl["category"])
                            registered += 1
                        except Exception:
                            pass
                    if registered > 0:
                        logger.info(f"[Startup] 注册 {registered} 个内置模板因子到空池")
            except Exception as e:
                logger.warning(f"[Startup] 内置因子注册跳过: {e}")
                all_ok = False
            try:
                from core.domain.factors.codegen import load_factors_from_db
                m = load_factors_from_db()
                if m > 0:
                    logger.info(f"[Startup] 从数据库恢复 {m} 个自定义因子")
            except Exception as e:
                logger.warning(f"[Startup] 因子恢复跳过: {e}")
                all_ok = False
            try:
                from core.domain.factors.registry import get_pool
                n = get_pool().register_fundamental_factors()
                if n > 0:
                    logger.info(f"[Startup] 注册 {n} 个基本面因子到因子池")
            except Exception as e:
                logger.warning(f"[Startup] 基本面因子注册跳过: {e}")
                all_ok = False
            _update_component("factors", "ok" if all_ok else "error",
                              "" if all_ok else "部分因子恢复失败")

        def _init_baostock_industry():
            try:
                from core.data.fetcher import DataFetcher
                fetcher = DataFetcher()
                ind_map = fetcher.get_industry_map()
                if ind_map:
                    _update_component("baostock", "ok")
                    logger.info(f"BaoStock 行业映射已缓存 ({len(ind_map)}只)")
                else:
                    _update_component("baostock", "error", "映射结果为空")
            except Exception as e:
                _update_component("baostock", "error", str(e))
                logger.warning(f"BaoStock 行业映射初始化失败: {e}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(thread_pool.default.raw_executor, _init_factors)
        await loop.run_in_executor(thread_pool.default.raw_executor, _init_baostock_industry)

        thread_pool.light.submit(_ensure_schedulers)

    _bg_task = asyncio.create_task(_delayed_init())

    def _on_delayed_done(task):
        try:
            exc = task.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                logger.error(f"[Startup] 后台初始化异常: {exc}", exc_info=True)
        except asyncio.CancelledError:
            pass

    _bg_task.add_done_callback(_on_delayed_done)

    s = thread_pool.stats()
    logger.info(f"[Startup] 线程池就绪: workers={s['total_workers']} health={s['health']}")

    yield  # ═══ 服务运行中 ═══

    # ═══ shutdown: 优雅关闭 ═══
    logger.info("[Shutdown] 开始优雅关闭...")

    # 1. 取消延迟初始化任务
    _bg_task.cancel()

    # 2. 停止调度器 (不接受新任务)
    try:
        await asyncio.get_running_loop().run_in_executor(thread_pool.default.raw_executor, _stop_schedulers)
    except Exception as e:
        logger.warning(f"[Shutdown] 调度器停止异常: {e}")

    # 3. 等待进行中的异步任务完成 (drained shutdown)
    await asyncio.sleep(3)

    # 4. 检查所有启动 Future 是否有未上报的异常
    for name, f in _startup_futures:
        try:
            exc = f.exception(timeout=2)
            if exc is not None:
                logger.error(f"[Startup] 启动任务 [{name}] 异常: {exc}", exc_info=exc)
        except Exception:
            pass

    # 5. 关闭线程池
    logger.info("[Shutdown] 关闭线程池...")
    thread_pool.shutdown()
    logger.info("[Shutdown] 线程池已关闭")


app = FastAPI(
    title="Apallg投研 - A股量化分析",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    lifespan=lifespan,
)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5174,http://localhost:5173,http://127.0.0.1:5174").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ═══ JWT 鉴权中间件 ═══
from backend.api.auth import decode_token, AUTH_WHITELIST

@app.middleware("http")
async def jwt_auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in AUTH_WHITELIST and path.rstrip("/") not in AUTH_WHITELIST:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if not token or decode_token(token) is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "未登录或登录已过期，请重新登录"},
            )
    return await call_next(request)

from backend.middleware import register_middleware
register_middleware(app)

import backend.api.alerts as alerts
import backend.api.auth as auth_api
import backend.api.backtest as backtest_api
import backend.api.factor as factor_api
import backend.api.macro as macro_api
import backend.api.market as market
import backend.api.news as news
import backend.api.portfolio as portfolio
import backend.api.data_manager as data_manager_api
import backend.api.qlib as qlib_api
import backend.api.regime as regime_api
import backend.api.scheduler as scheduler_api
import backend.api.screening as screening
import backend.api.sectors as sectors
import backend.api.sentiment as sentiment
import backend.api.settings as settings_api
import backend.api.strategy_profiles as strategy_profiles_api
import backend.api.stock as stock
import backend.api.system as system_api
import backend.api.ths as ths
import backend.api.timing as timing
import backend.api.trading as trading
import backend.api.tuner as tuner
import backend.api.rebalance as rebalance_api

app.include_router(auth_api.router, prefix="/api", tags=["认证"])
app.include_router(market.router, prefix="/api/market", tags=["大盘"])
app.include_router(stock.router, prefix="/api/stock", tags=["个股"])
app.include_router(ths.router)
app.include_router(sectors.router, prefix="/api/sectors", tags=["板块"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["持仓"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["告警"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["情绪"])
app.include_router(news.router, prefix="/api/news", tags=["新闻"])
app.include_router(trading.router, prefix="/api/trading", tags=["交易"])
app.include_router(scheduler_api.router, prefix="/api/scheduler", tags=["定时扫描"])
app.include_router(data_manager_api.router, prefix="/api/data", tags=["数据中心"])
app.include_router(qlib_api.router, prefix="/api/qlib", tags=["Qlib ML"])
app.include_router(screening.router, prefix="/api/screening", tags=["选股筛选"])
app.include_router(rebalance_api.router, prefix="/api/rebalance", tags=["调仓信号"])
app.include_router(tuner.router, prefix="/api/tuner", tags=["配置调优"])
app.include_router(regime_api.router, prefix="/api/regime", tags=["市场状态"])
app.include_router(factor_api.router, prefix="/api/factor", tags=["因子管理"])
app.include_router(factor_api.router, prefix="/api/whitebox", tags=["白盒因子"])
app.include_router(system_api.router, prefix="/api/system", tags=["系统管理"])
app.include_router(macro_api.router, prefix="/api/macro", tags=["宏观数据"])
app.include_router(timing.router, prefix="/api/timing", tags=["择时模型"])
app.include_router(backtest_api.router, prefix="/api/backtest", tags=["策略回测"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["系统设置"])
app.include_router(strategy_profiles_api.router, prefix="/api/strategy-profiles", tags=["策略版本"])

# ═══ 生产模式: 后端 serve 前端静态文件 ═══
_SERVE_STATIC = os.getenv("SERVE_STATIC", "false").lower() == "true"
if _SERVE_STATIC:
    _static_dir = Path(__file__).parent.parent / "frontend" / "dist"
    if _static_dir.exists():
        from fastapi.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="static_assets")
        _index_path = _static_dir / "index.html"

        @app.middleware("http")
        async def _spa_fallback(request: Request, call_next):
            response = await call_next(request)
            if response.status_code == 404 and not request.url.path.startswith("/api"):
                return FileResponse(str(_index_path))
            return response

        logger.info(f"[Static] 前端静态文件已挂载: {_static_dir} (SPA fallback)")


# ═══ 分级健康检查 ═══
@app.get("/api/health")
def health():
    errors = []
    degraded = []

    with _components_lock:
        snapshot = {
            name: {"status": c["status"], "importance": c["importance"], "detail": c.get("detail", "")}
            for name, c in _components.items()
        }

    if snapshot.get("qlib", {}).get("status") == "ok" and snapshot.get("db", {}).get("status") == "ok":
        try:
            from backend.service.market_data import MarketDataService
            decision_date = MarketDataService.get_screening_decision_date()
            qlib_freshness = MarketDataService.get_qlib_readiness(decision_date)
            if not qlib_freshness["ready"]:
                if (not qlib_freshness.get("latest_date")
                        or qlib_freshness["latest_date"] < qlib_freshness["decision_date"]):
                    detail = (
                        f"Qlib数据仅到 {qlib_freshness['latest_date'] or '未知'}，"
                        f"落后交易日 {qlib_freshness['decision_date']}"
                    )
                else:
                    detail = (
                        f"Qlib决策日截面不完整: {qlib_freshness['available']}/"
                        f"{qlib_freshness['expected']}"
                    )
                snapshot["qlib"]["status"] = "error"
                snapshot["qlib"]["detail"] = detail
                degraded.append({"component": "qlib", "detail": detail})
        except Exception as e:
            detail = f"Qlib新鲜度检查失败: {e}"
            snapshot["qlib"]["status"] = "error"
            snapshot["qlib"]["detail"] = detail
            degraded.append({"component": "qlib", "detail": detail})

    for name, comp in snapshot.items():
        imp = comp["importance"]
        st = comp["status"]
        if st == "error" and imp == "required":
            errors.append({"component": name, "detail": comp["detail"]})
        elif st in ("error", "pending") and imp == "preferred":
            degraded.append({"component": name, "detail": comp["detail"]})
        elif st == "pending" and imp == "required":
            degraded.append({"component": name, "detail": "等待初始化"})

    if errors:
        status = "fatal"
        summary = f"{len(errors)} 个关键组件异常"
    elif degraded:
        status = "degraded"
        summary = f"{len(degraded)} 个组件降级或等待中"
    else:
        status = "ok"
        summary = "所有组件正常"

    return {
        "status": status,
        "summary": summary,
        "app": "Apallg投研",
        "version": "1.0.0",
        "components": snapshot,
        "schedulers": _schedulers_started,
        "qlib_ready": _qlib_ready.is_set(),
        "serve_static": _SERVE_STATIC,
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 Apallg投研 API 服务器 http://0.0.0.0:8001")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=False)
