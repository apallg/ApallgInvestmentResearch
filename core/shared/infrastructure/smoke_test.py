"""全项目冒烟测试——每项独立 try/except + 线程级超时"""
import time
import datetime
import json
import logging

from core.shared import config

logger = logging.getLogger(__name__)


class SkipCheck(Exception):
    """检查项主动跳过（如组件未启用）"""


class SmokeTestRunner:
    def __init__(self):
        self.checks = [
            ("MySQL", self._check_mysql, None),
            ("Redis", self._check_redis, None),
            ("diskcache", self._check_diskcache, None),
            ("通知通道", self._check_notifier, None),
            ("因子池", self._check_factor_pool, None),
            ("FakeBroker", self._check_fake_broker, None),
            ("日K线", self._check_stock_daily, None),
            ("指数日K", self._check_index_daily, None),
            ("实时行情", self._check_realtime, None),
            ("择时模型", self._check_timing, None),
            ("市场状态", self._check_regime, None),
            ("Qlib模型", lambda: self._run_qlib_read_check(self._check_qlib_model), None),
            ("情绪模型", self._check_sentiment, 120),
            ("选股快照", lambda: self._run_qlib_read_check(self._check_snapshot), 120),
            ("数据完整性", self._check_data_integrity, 120),
        ]

    def run_all(self, source="scheduled"):
        start = time.time()
        default_timeout = getattr(config, "SMOKE_TEST_ITEM_TIMEOUT", 60)
        results = []
        for name, fn, timeout in self.checks:
            r = self._run_one(name, fn, timeout or default_timeout)
            logger.info(f"[冒烟] {name}: {r['status']} ({r['elapsed']}s) {r.get('error') or r.get('detail')}")
            results.append(r)
        report = {
            "run_date": str(datetime.date.today()),
            "source": source,
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "pass"),
            "failed": sum(1 for r in results if r["status"] == "fail"),
            "skipped": sum(1 for r in results if r["status"] == "skip"),
            "elapsed_seconds": round(time.time() - start, 1),
            "results": results,
        }
        report["status"] = "success" if report["failed"] == 0 else "partial"
        self._log_report(report)
        if getattr(config, "SMOKE_TEST_NOTIFY", True):
            self._notify(report)
        return report

    def _run_one(self, name, fn, timeout):
        from core.shared.infrastructure.thread_pool import thread_pool

        holder = {}

        def _target():
            try:
                holder["detail"] = fn() or ""
                holder["status"] = "pass"
            except SkipCheck as e:
                holder["status"] = "skip"
                holder["error"] = str(e)
            except Exception as e:
                holder["status"] = "fail"
                holder["error"] = f"{type(e).__name__}: {e}"

        future = thread_pool.light.submit(_target)
        s = time.time()
        try:
            future.result(timeout=timeout)
        except Exception:
            pass
        elapsed = round(time.time() - s, 1)
        if not future.done():
            return {"name": name, "status": "fail", "elapsed": elapsed,
                    "error": f"超时(>{timeout}s)", "detail": ""}
        return {"name": name, "status": holder.get("status", "fail"),
                "elapsed": elapsed, "error": holder.get("error", ""),
                "detail": str(holder.get("detail", ""))}

    def _run_qlib_read_check(self, check):
        from backend.service.qlib_access import run_with_qlib_read_access

        return run_with_qlib_read_access(check)

    # ── 检查项 ──

    def _check_mysql(self):
        from core.data import Database
        if not Database.is_available():
            raise SkipCheck("MYSQL_ENABLED=false")
        row = Database.fetchone("SELECT 1 AS ok")
        assert row, "SELECT 1 无返回"
        return "连接正常"

    def _check_redis(self):
        from utils import redis_client
        if not getattr(config, "REDIS_ENABLED", True):
            raise SkipCheck("REDIS_ENABLED=false")
        assert redis_client.redis_enabled(), "Redis 不可达"
        return "连接正常"

    def _check_diskcache(self):
        from core.shared.infrastructure.cache import CacheManager
        cache = CacheManager()
        cache.set("smoke_test_probe", "1", expire=60)
        assert cache.get("smoke_test_probe") == "1", "读写不一致"
        return "读写正常"

    def _check_notifier(self):
        from core.shared.infrastructure.notify import get_channel_status
        status = get_channel_status()
        if not any(status.values()):
            raise SkipCheck("未配置任何通知通道")
        return ", ".join(k for k, v in status.items() if v)

    def _check_factor_pool(self):
        from core.domain.factors.registry import get_pool
        factors = get_pool().get_active_factors()
        assert factors, "活跃因子为空"
        return f"{len(factors)} 个活跃因子"

    def _check_fake_broker(self):
        from core.adapters.broker import create_broker
        b = create_broker("fake")
        assert b.connect(), "connect 失败"
        b.get_account()
        b.get_positions()
        b.disconnect()
        return "只读链路正常"

    def _check_stock_daily(self):
        from core.data.fetcher import DataFetcher
        code = getattr(config, "SMOKE_TEST_BACKTEST_CODE", "600519")
        df = DataFetcher().get_stock_daily(code, days=30)
        assert df is not None and not df.empty, "日K线为空"
        return f"{code} {len(df)} 条"

    def _check_index_daily(self):
        from core.data.fetcher import DataFetcher
        df = DataFetcher().get_index_daily("上证指数", days=60)
        assert df is not None and not df.empty, "指数日K为空"
        return f"{len(df)} 条"

    def _check_realtime(self):
        from core.data.fetcher import DataFetcher
        from core.shared.trading_time import is_trading_time
        code = getattr(config, "SMOKE_TEST_BACKTEST_CODE", "600519")
        quote = DataFetcher().get_realtime_quote(code)
        if not quote:
            if not is_trading_time():
                raise SkipCheck("非交易时段，实时数据源不可用")
            raise AssertionError("实时行情为空")
        return "行情正常"

    def _check_timing(self):
        from core.domain.timing.model import get_timing_ratio
        ratio = get_timing_ratio()
        assert ratio is not None, "择时模型未训练或预测失败"
        return f"仓位比例 {ratio}"

    def _check_regime(self):
        from core.domain.regime.classifier import MarketRegimeClassifier
        result = MarketRegimeClassifier().classify()
        err = getattr(result, "details", {}).get("error") if getattr(result, "details", None) else None
        assert not err, f"分类失败: {err}"
        return f"{getattr(result, 'regime', result)}"

    def _check_qlib_model(self):
        from core.domain.screening._screener import BuySideScreener
        model = BuySideScreener()._find_best_qlib_model()
        assert model, "mlruns 中无可用 Qlib 模型"
        return f"exp={model[0]} run={model[1][:8]}"

    def _check_sentiment(self):
        from core.adapters.nlp.analyzer import SentimentAnalyzer
        r = SentimentAnalyzer().analyze([{"title": "今天股市大涨"}])
        assert r and r[0].get("sentiment") in ("positive", "negative", "neutral"), f"结果异常: {r}"
        return f"{r[0]['sentiment']}"

    def _check_snapshot(self):
        from core.domain.screening._screener import BuySideScreener
        code = getattr(config, "SMOKE_TEST_BACKTEST_CODE", "600519")
        snap = BuySideScreener().precompute_snapshot(code_prefixes=[code], skip_tech=True)
        assert snap, "快照为空"
        return f"{len(snap)} 只"

    def _check_data_integrity(self):
        from core.data import Database
        if not Database.is_available():
            raise SkipCheck("MYSQL_ENABLED=false")

        ref = Database.fetchone("SELECT MAX(trade_date) AS d FROM daily_quotes")
        if not ref or not ref["d"]:
            raise SkipCheck("daily_quotes 无数据，跳过完整性检查")
        ref_date = ref["d"]

        tables = [
            ("daily_quotes", "trade_date", "日频", 1),
            ("daily_basic", "trade_date", "日频", 2),
            ("north_flow", "trade_date", "日频", 2),
            ("capital_flow_daily", "trade_date", "日频", 2),
            ("sentiment_daily", "trade_date", "日频", 2),
            ("daily_forward_ret", "trade_date", "日频", 2),
            ("etf_fund_flow", "trade_date", "日频", 2),
            ("macro_shibor", "trade_date", "日频", 2),
            ("macro_lpr", "trade_date", "月频", 40),
            ("macro_money_supply", "trade_date", "月频", 55),
            ("macro_cpi", "trade_date", "月频", 55),
            ("macro_ppi", "trade_date", "月频", 55),
            ("macro_pmi", "trade_date", "月频", 55),
            ("fundamental_data", "report_date", "季频", 130),
            ("forecast_data", "end_date", "季频", 130),
            ("income_data", "end_date", "季频", 130),
            ("analyst_forecast", "report_date", "季频", 130),
        ]

        exist = Database.fetchall(
            "SELECT TABLE_NAME FROM information_schema.tables "
            "WHERE table_schema = DATABASE()"
        )
        exist_set = {r["TABLE_NAME"] for r in exist}

        issues = []
        ok_count = 0
        for table, date_col, freq, max_days in tables:
            if table not in exist_set:
                continue
            row = Database.fetchone(
                f"SELECT MAX(`{date_col}`) AS max_d, COUNT(*) AS cnt FROM `{table}`"
            )
            max_d = row["max_d"] if row else None
            cnt = row["cnt"] if row else 0

            if max_d is None or cnt == 0:
                issues.append(f"{table}: 空表")
                continue

            days_behind = (ref_date - max_d).days
            if days_behind > max_days:
                issues.append(
                    f"{table}({freq}): 滞后{days_behind}天 最新{max_d} 参考{ref_date}"
                )
            else:
                ok_count += 1

        if issues:
            raise AssertionError(f"数据滞后({ref_date}): {'; '.join(issues)}")
        return f"{ok_count}个表正常, 参考={ref_date}"

    # ── 持久化与通知 ──

    def _log_report(self, report):
        try:
            from core.data import Database
            if not Database.is_available():
                return
            Database.execute(
                "INSERT INTO smoke_test_log "
                "(run_date, source, total, passed, failed, skipped, elapsed_seconds, status, detail) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    report["run_date"],
                    report["source"],
                    report["total"],
                    report["passed"],
                    report["failed"],
                    report["skipped"],
                    report["elapsed_seconds"],
                    report["status"],
                    json.dumps(report["results"], ensure_ascii=False),
                ),
            )
        except Exception as e:
            logger.warning(f"冒烟测试结果写入失败: {e}")

    def _notify(self, report):
        try:
            from core.shared.infrastructure.notify import get_wecom, _safe_truncate_bytes, WECOM_MARKDOWN_MAX_BYTES
            n = get_wecom()
            if not n.enabled:
                return
            d = report["run_date"]
            if report["failed"] == 0:
                slowest = max(report["results"], key=lambda r: r["elapsed"])
                content = (
                    f"## 每日冒烟测试全部通过 {d}\n"
                    f"> **{report['passed']}/{report['total']} 通过** | "
                    f"总耗时 {report['elapsed_seconds']}s | 最慢: {slowest['name']}({slowest['elapsed']}s)"
                )
            else:
                fails = [r for r in report["results"] if r["status"] == "fail"]
                lines = [
                    f"## 每日冒烟测试 {len(fails)} 项失败 {d}",
                    f"> **{report['passed']} 通过 / {report['failed']} 失败 / {report['skipped']} 跳过** | 总耗时 {report['elapsed_seconds']}s",
                    "**失败详情**",
                ]
                per_budget = max(200, 2800 // max(len(fails), 1))
                for r in fails:
                    lines.append(
                        f"> <font color=\"warning\">{r['name']}</font> ({r['elapsed']}s): "
                        + _safe_truncate_bytes(r["error"], per_budget))
                passed = "、".join(r["name"] for r in report["results"] if r["status"] == "pass")
                lines.append(f"**通过**: {_safe_truncate_bytes(passed, 600)}")
                content = "\n".join(lines)
            n.send_markdown(_safe_truncate_bytes(content, WECOM_MARKDOWN_MAX_BYTES))
        except Exception as e:
            logger.warning(f"冒烟测试通知失败: {e}")


def get_smoke_history(limit=30):
    from core.data import Database
    if not Database.is_available():
        return []
    try:
        rows = Database.fetchall(
            "SELECT id, run_date, source, total, passed, failed, skipped, "
            "elapsed_seconds, status, created_at FROM smoke_test_log "
            "ORDER BY id DESC LIMIT %s", (limit,))
        for r in rows:
            if r.get("run_date"):
                r["run_date"] = str(r["run_date"])
            if r.get("created_at"):
                r["created_at"] = str(r["created_at"])
            if r.get("elapsed_seconds") is not None:
                r["elapsed_seconds"] = float(r["elapsed_seconds"])
        return rows
    except Exception as e:
        logger.warning(f"冒烟测试历史查询失败: {e}")
        return []
