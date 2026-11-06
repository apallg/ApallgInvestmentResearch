"""数据质量服务 — 数据完整性/检查/修复编排。"""
import json
import logging
import threading
import uuid
from concurrent.futures import as_completed

from core.domain.screening.exclusion import is_excluded
from core.data.sources.tencent_kline import fetch_tencent_kline_raw
from core.data.sources.qfq_normalizer import prepare_qfq_daily_frame
from core.shared.infrastructure.thread_pool import thread_pool
from core.shared import config
from backend.service.qlib_access import qlib_read_locked

logger = logging.getLogger(__name__)

_jobs = {}
_jobs_lock = threading.Lock()


def _set_job(job_id, data):
    with _jobs_lock:
        _jobs[job_id] = data
    try:
        import redis
        _r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
        _r.setex(f"qlib:job:{job_id}", 3600, json.dumps(data, ensure_ascii=False, default=str))
    except Exception:
        pass


def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        return job
    try:
        import redis
        _r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2, decode_responses=True)
        raw = _r.get(f"qlib:job:{job_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


class DataQualityService:

    @staticmethod
    def data_quality():
        """检查每日数据完整性 — 按股票统计覆盖范围"""
        import datetime
        from core.data import Database

        today = datetime.date.today()
        ref = Database.fetchone("SELECT COUNT(*) as cnt FROM daily_quotes WHERE code = '600519'")["cnt"]

        rows = Database.fetchall("""
            SELECT code, MIN(trade_date) as mind, MAX(trade_date) as maxd, COUNT(*) as cnt
            FROM daily_quotes GROUP BY code ORDER BY cnt DESC
        """)
        stocks = []
        zero_count = 0
        short_count = 0
        good_count = 0
        total = 0

        for row in rows:
            code = row["code"]
            mind = row["mind"]
            maxd = row["maxd"]
            cnt = row["cnt"]
            total += 1

            if cnt == 0:
                zero_count += 1
                status = "empty"
            elif cnt < 100:
                short_count += 1
                status = "short"
            elif cnt >= 500:
                good_count += 1
                status = "good"
            else:
                status = "ok"

            stocks.append({
                "code": code,
                "first_date": str(mind) if mind else None,
                "last_date": str(maxd) if maxd else None,
                "records": cnt,
                "status": status,
            })

        missing_codes = []
        try:
            from concurrent.futures import TimeoutError as FutureTimeoutError
            from core.shared.infrastructure.thread_pool import thread_pool

            def _get_codes():
                import akshare as ak
                return __import__('core.data.sources.tushare', fromlist=['get_all_stock_codes']).get_all_stock_codes()

            all_codes = thread_pool.default.submit(_get_codes).result(timeout=5)
            existing = {s["code"] for s in stocks}
            missing_codes = [c for c in all_codes if c not in existing
                             and not is_excluded(c)[0]]
        except (FutureTimeoutError, Exception):
            pass

        return {"data": {
            "total_stocks": total,
            "missing_stocks": len(missing_codes),
            "zero_records": zero_count,
            "short_records": short_count,
            "good_records": good_count,
            "stocks": stocks[:100],
            "worst": [s for s in stocks if s["status"] in ("empty", "short")][:20],
            "missing_sample": missing_codes[:20],
        }}

    @staticmethod
    @qlib_read_locked
    def validate_qlib_data():
        """验证 Qlib 二进制数据是否正常（防止标签全0等问题）"""
        from pathlib import Path
        from core.adapters.qlib.generation import assert_no_active_historical_release

        assert_no_active_historical_release(config.QLIB_DATA_ROOT)

        result = {"checks": [], "passed": True}

        import numpy as np
        data_dir = Path(config.QLIB_DATA_DIR)

        # 1. 日历
        cal_path = data_dir / "calendars" / "day.txt"
        if not cal_path.exists():
            result["checks"].append({"item": "日历文件", "ok": False, "msg": "day.txt 不存在"})
            result["passed"] = False
            return {"data": result}
        dates = cal_path.read_text().strip().splitlines()
        result["checks"].append({"item": "日历", "ok": True, "msg": f"{len(dates)} 天 ({dates[0]}~{dates[-1]})"})

        # 2. Instruments
        inst_dir = data_dir / "instruments"
        inst_files = list(inst_dir.glob("*.txt"))
        for f in inst_files:
            n = len(f.read_text().strip().splitlines())
            result["checks"].append({"item": f"instruments/{f.name}", "ok": n > 0, "msg": f"{n} 只"})
            if n == 0:
                result["passed"] = False

        # 3. 抽样检查 feature bin 文件
        features_dir = data_dir / "features"
        sample_codes = ["sh600519", "sz000001", "sh600036", "sz000858"]
        for code in sample_codes:
            stock_dir = features_dir / code
            if not stock_dir.exists():
                result["checks"].append({"item": f"features/{code}", "ok": False, "msg": "目录不存在"})
                result["passed"] = False
                continue
            issues = []
            for field in ["close", "open", "high", "low", "volume"]:
                bin_path = stock_dir / f"{field}.day.bin"
                if not bin_path.exists():
                    issues.append(f"{field}.bin 缺失")
                    continue
                try:
                    data = np.fromfile(str(bin_path), dtype="<f")
                    if len(data) < 2:
                        issues.append(f"{field}.bin 空文件")
                        continue
                    start_idx = int(data[0])
                    values = data[1:]
                    non_nan = values[~np.isnan(values)]
                    if len(non_nan) == 0:
                        issues.append(f"{field} 全 NaN")
                    elif np.all(non_nan == 0):
                        issues.append(f"{field} 全 0")
                except Exception as e:
                    issues.append(f"{field}.bin 读取失败: {e}")
            if issues:
                result["checks"].append({"item": f"features/{code}", "ok": False, "msg": "; ".join(issues)})
                result["passed"] = False
            else:
                n_vals = len(np.fromfile(str(stock_dir / "close.day.bin"), dtype="<f")) - 1
                result["checks"].append({"item": f"features/{code}", "ok": True, "msg": f"{n_vals} 条, 数据正常"})

        return {"data": result}

    @staticmethod
    def check_data_gaps(lookback_days: int = 5, limit: int = 200):
        """检测近期有数据缺口的股票（有某些交易日但缺中间某天）"""
        from scripts.fast_collect import detect_gaps
        gaps = detect_gaps(lookback_days)
        gap_list = [
            {"code": code, "missing_dates": missing, "gap_count": len(missing)}
            for code, missing in sorted(gaps.items(), key=lambda x: -len(x[1]))[:limit]
        ]
        return {
            "data": {
                "gaps": gap_list,
                "total_gap_stocks": len(gaps),
                "lookback_days": lookback_days,
                "sample_dates": list(list(gaps.values())[0]) if gaps else [],
            }
        }

    @staticmethod
    def data_check():
        """综合数据检查：缺口 + 单位一致性 + 覆盖度"""
        import datetime
        from core.data import Database

        result = {
            "quality": {},
            "gaps": {"total_gap_stocks": 0, "gap_list": [], "sample_dates": []},
            "unit_warnings": [],
            "checks": [],
        }

        today = datetime.date.today()
        rows = Database.fetchall("""
            SELECT code, MIN(trade_date) as mind, MAX(trade_date) as maxd, COUNT(*) as cnt
            FROM daily_quotes GROUP BY code ORDER BY cnt DESC
        """)
        total = len(rows)
        good_count = sum(1 for r in rows if r["cnt"] >= 500)
        short_count = sum(1 for r in rows if r["cnt"] < 100 and r["cnt"] > 0)
        zero_count = sum(1 for r in rows if r["cnt"] == 0)

        result["quality"] = {
            "total_stocks": total,
            "good_records": good_count,
            "short_records": short_count,
            "zero_records": zero_count,
        }
        result["checks"].append(f"覆盖度: {total}只, ≥500条 {good_count}, <100条 {short_count}")

        try:
            date_rows = Database.fetchall(
                "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
            )
            if date_rows:
                all_dates = sorted(set(
                    str(r["trade_date"].date() if hasattr(r["trade_date"], "date") else r["trade_date"])
                    for r in date_rows
                ))

                ref = Database.fetchall(
                    "SELECT code, trade_date FROM daily_quotes "
                    "WHERE code IN ('600519','000001','300750','000858') ORDER BY code, trade_date"
                )
                ref_by_code = {}
                for r in ref:
                    ref_by_code.setdefault(r["code"], set()).add(
                        str(r["trade_date"].date() if hasattr(r["trade_date"], "date") else r["trade_date"])
                    )
                sample_gaps = []
                for code, dates in ref_by_code.items():
                    recent_window = all_dates[-60:] if len(all_dates) >= 60 else all_dates
                    missing = [d for d in recent_window if d not in dates]
                    if missing:
                        sample_gaps.append({
                            "code": code, "recent_missing": len(missing),
                            "first_missing": missing[0], "last_missing": missing[-1],
                        })
                if sample_gaps:
                    result["gaps"]["sample_check"] = sample_gaps
                    result["checks"].append(
                        "日期缺口(抽样): " +
                        ", ".join(f"{g['code']}缺{g['recent_missing']}天" for g in sample_gaps)
                    )

                from scripts.fast_collect import detect_gaps
                recent_gaps = detect_gaps(lookback_days=30)
                gap_list = [
                    {"code": code, "missing_dates": missing[:10], "gap_count": len(missing)}
                    for code, missing in sorted(recent_gaps.items(), key=lambda x: -len(x[1]))[:50]
                ]
                result["gaps"]["total_gap_stocks"] = len(recent_gaps)
                result["gaps"]["gap_list"] = gap_list
                result["gaps"]["lookback_days"] = 30
                if gap_list:
                    result["gaps"]["sample_dates"] = gap_list[0]["missing_dates"]
                result["checks"].append(f"近30天缺口: {len(recent_gaps)}只有缺失")

                # 全历史覆盖度: 用二分查找代替 O(N*M) 线性扫描
                if all_dates:
                    import bisect
                    low_coverage = []
                    for r in rows:
                        code = r["code"]
                        if is_excluded(code)[0]:
                            continue
                        mind = str(r["mind"].date() if hasattr(r["mind"], "date") else r["mind"])
                        maxd = str(r["maxd"].date() if hasattr(r["maxd"], "date") else r["maxd"])
                        left = bisect.bisect_left(all_dates, mind)
                        right = bisect.bisect_right(all_dates, maxd)
                        expected = right - left
                        actual = r["cnt"]
                        if expected > 50 and actual / expected < 0.85:
                            low_coverage.append((code, actual, expected, round(actual/expected*100)))
                    low_coverage.sort(key=lambda x: x[3])
                    if low_coverage:
                        result["gaps"]["low_coverage"] = [
                            {"code": c, "actual": a, "expected": e, "pct": p}
                            for c, a, e, p in low_coverage[:30]
                        ]
                        result["checks"].append(
                            f"全历史覆盖不足(85%): {len(low_coverage)}只 "
                            f"(最差: {low_coverage[0][0]} {low_coverage[0][3]}%)"
                        )
                    else:
                        result["checks"].append("全历史覆盖度: 正常")
        except Exception as e:
            result["checks"].append(f"缺口检测失败: {e}")

        try:
            suspect = Database.fetchall("""
                SELECT code, AVG(volume) as avg_vol, AVG(amount) as avg_amt,
                       AVG(amount) / NULLIF(AVG(volume), 0) as avg_price,
                       AVG(close) as avg_close, COUNT(*) as cnt
                FROM daily_quotes WHERE volume > 0 AND amount > 0
                GROUP BY code
                HAVING avg_price < 1 OR avg_price > 5000
                ORDER BY avg_price
                LIMIT 50
            """)
            # 区分：真异常（均价偏离收盘均价太多） vs 自然低价/高价股
            real_unit_issues = []
            for r in suspect:
                code = r["code"]
                # ETF/基金 价格低于1元是正常的，跳过
                if is_excluded(code)[0]:
                    continue
                avg_close = float(r["avg_close"] or 0)
                avg_price = float(r["avg_price"] or 0)
                ratio = abs(avg_price - avg_close) / max(avg_close, 0.01)
                if ratio > 0.1 and avg_price < avg_close * 0.5:
                    real_unit_issues.append({
                        "code": r["code"], "avg_vol": round(r["avg_vol"], 0),
                        "avg_amt": round(r["avg_amt"], 0), "avg_price": round(avg_price, 2),
                        "avg_close": round(avg_close, 2), "records": r["cnt"],
                        "reason": "量价背离(volume可能未转股)"
                    })
                elif avg_price < 1:
                    real_unit_issues.append({
                        "code": r["code"], "avg_vol": round(r["avg_vol"], 0),
                        "avg_amt": round(r["avg_amt"], 0), "avg_price": round(avg_price, 2),
                        "avg_close": round(avg_close, 2), "records": r["cnt"],
                        "reason": "均价<1元(可能单位残留)"
                    })
            if real_unit_issues:
                result["unit_warnings"] = real_unit_issues
                result["checks"].append(
                    f"单位异常: {len(real_unit_issues)}只 "
                    f"(均价范围 {min(r['avg_price'] for r in real_unit_issues):.2f}~{max(r['avg_price'] for r in real_unit_issues):.2f})"
                )
            else:
                result["checks"].append("单位检查: 未发现单位异常")
        except Exception as e:
            result["checks"].append(f"单位检查失败: {e}")

        return {"data": result}

    @staticmethod
    def start_fix_units(background_tasks=None):
        """检测并修复单位异常的股票（删旧数据 → Tencent重采）"""
        from core.data import Database, DailyQuotesRepo
        import threading as th
        import pandas as pd

        suspect = Database.fetchall("""
            SELECT code, AVG(amount)/NULLIF(AVG(volume),0) as avg_price, AVG(close) as avg_close
            FROM daily_quotes WHERE volume > 0 AND amount > 0
            GROUP BY code
            HAVING (AVG(amount)/NULLIF(AVG(volume),0) < 1
                    OR AVG(amount)/NULLIF(AVG(volume),0) > 5000)
        """)
        codes = []
        skipped = 0
        for r in suspect:
            code = r["code"]
            if is_excluded(code)[0]:
                skipped += 1
                continue
            avg_close = float(r["avg_close"] or 0)
            avg_price = float(r["avg_price"] or 0)
            ratio = abs(avg_price - avg_close) / max(avg_close, 0.01)
            if ratio > 0.1 and avg_price < avg_close * 0.5:
                codes.append(r["code"])
            elif avg_price < 1:
                codes.append(r["code"])
            else:
                skipped += 1
        total_flagged = len(suspect) - skipped
        if not codes:
            return {"data": {"status": "completed", "message": "未发现单位异常的股票"}}

        job_id = str(uuid.uuid4())[:8]
        _set_job(job_id, {"status": "running", "message": f"删除 {len(codes)} 只异常数据..."})

        def _do_fix():
            total = len(codes)
            success = [0]
            fail = [0]
            lock = th.Lock()
            err_samples = []

            _set_job(job_id, {"status": "running", "message": f"开始安全重采 {total} 只；候选行情通过校验后才原子替换"})

            def _fetch_and_save(code):
                try:
                    _, rows_list = fetch_tencent_kline_raw(code, adj="", timeout=15)
                    if not rows_list:
                        raise RuntimeError("无有效K线")
                    df = pd.DataFrame(rows_list)
                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                    df["amplitude"] = None
                    df["turnover"] = None
                    DailyQuotesRepo.save_batch(code, prepare_qfq_daily_frame(code, df, source="tencent", task_id=job_id), replace_dates=True)
                    with lock:
                        success[0] += 1
                except Exception as e:
                    with lock:
                        fail[0] += 1
                        if len(err_samples) < 5:
                            err_samples.append(f"{code}: {e}")

            with thread_pool.default.borrow() as ex:
                futs = {ex.submit(_fetch_and_save, c): c for c in codes}
                for i, f in enumerate(as_completed(futs)):
                    f.result()
                    if (i + 1) % 50 == 0 or i + 1 == total:
                        _set_job(job_id, {
                            "status": "running",
                            "message": f"修复进度: {i+1}/{total}, 成功 {success[0]}, 失败 {fail[0]}"
                        })

            msg = f"修复完成: {success[0]} 成功, {fail[0]} 失败"
            if err_samples:
                msg += f" | 错误: {'; '.join(err_samples[:3])}"
            _set_job(job_id, {"status": "completed", "message": msg})

        if background_tasks is not None:
            background_tasks.add_task(_do_fix)
        else:
            _do_fix()
        return {"data": {"job_id": job_id, "status": "started",
                        "message": f"开始修复 {len(codes)} 只单位异常股票"}}
