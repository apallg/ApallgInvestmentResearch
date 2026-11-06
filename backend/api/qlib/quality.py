"""DEPRECATED: 已由 backend/api/qlib/quality_v2.py 和 backend/service/data_quality.py 取代，请勿新增逻辑。"""
import anyio
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from core.data.sources.qfq_normalizer import prepare_qfq_daily_frame
from backend.api.qlib.common import (  # noqa: F403
    logger, _train_jobs, _sync_lock, _collect_job_id,
    _OPTUNA_DIR, _save_optuna_result, _load_optuna_result,
    _get_qlib_data_dir, _resolve_instruments, _build_filtered_instruments,
    _filter_st_stocks, _get_all_stock_codes, _resolve_factor_label,
    _make_display_name, _assert_run_path,
    is_excluded, EXCLUDED_PREFIXES, B_SHARE_PREFIXES,
    fetch_tencent_kline_raw, task_manager, thread_pool, uuid,
)

router = APIRouter()

@router.get("/data/quality")
async def data_quality():
    """检查每日数据完整性 — 按股票统计覆盖范围"""
    def _run():
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
    return await anyio.to_thread.run_sync(_run)
@router.get("/data/validate-qlib")
async def validate_qlib_data():
    """验证 Qlib 二进制数据是否正常（防止标签全0等问题）"""
    from backend.service.data_quality import DataQualityService

    return await anyio.to_thread.run_sync(DataQualityService.validate_qlib_data)
@router.get("/data/gaps")
async def check_data_gaps(lookback_days: int = 5, limit: int = 200):
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
@router.get("/data/check")
async def data_check():
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
@router.post("/data/fix-units")
async def fix_unit_issues(background_tasks: BackgroundTasks):
    """检测并修复单位异常的股票（删旧数据 → Tencent重采）"""
    from core.data import Database, DailyQuotesRepo
    from concurrent.futures import as_completed
    from core.shared.infrastructure.thread_pool import thread_pool
    import threading, requests as req_lib, pandas as pd

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
    _train_jobs[job_id] = {"status": "running", "message": f"删除 {len(codes)} 只异常数据..."}
    try:
        import redis, json
        _r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
        _r.setex(f"qlib:job:{job_id}", 3600, json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
    except Exception:
        pass

    def _do_fix():
        total = len(codes)
        success = [0]; fail = [0]
        lock = threading.Lock()
        err_samples = []

        _train_jobs[job_id]["message"] = f"开始安全重采 {total} 只；候选行情通过校验后才原子替换"

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
                    _train_jobs[job_id]["message"] = (
                        f"修复进度: {i+1}/{total}, 成功 {success[0]}, 失败 {fail[0]}"
                    )

        msg = f"修复完成: {success[0]} 成功, {fail[0]} 失败"
        if err_samples:
            msg += f" | 错误: {'; '.join(err_samples[:3])}"
        _train_jobs[job_id] = {"status": "completed", "message": msg}
        try:
            import redis, json
            _r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=2)
            _r.setex(f"qlib:job:{job_id}", 3600, json.dumps(_train_jobs[job_id], ensure_ascii=False, default=str))
        except Exception:
            pass

    background_tasks.add_task(_do_fix)
    return {"data": {"job_id": job_id, "status": "started",
                    "message": f"开始修复 {len(codes)} 只单位异常股票"}}


# ══════════════════════════════════════════════
# 新数据采集端点（拆分增量/全量）
# ══════════════════════════════════════════════
