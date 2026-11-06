"""
服务器极速数据采集脚本

腾讯 IFZQ HTTP → MySQL → qlib 同步 (默认, 32 线程并发, ~3 分钟)
BaoStock → MySQL → qlib 同步 (串行, 数据完整, ~50 分钟)
混合模式 (推荐): 腾讯先拿近2年 → BaoStock 补全历史

用法:
  python scripts/fast_collect.py                     # 腾讯源, 32线程
  python scripts/fast_collect.py --source hybrid      # 混合模式(推荐): 腾讯+补全历史
  python scripts/fast_collect.py --source hybrid --pool csi300
  python scripts/fast_collect.py --source baostock    # BaoStock 纯慢速完整采集
  python scripts/fast_collect.py --workers 64         # 腾讯源, 64线程
  python scripts/fast_collect.py --no-sync            # 只采不同步qlib
"""
import argparse
import datetime
import logging
import sys
import time
import threading
from concurrent.futures import as_completed
from core.shared.infrastructure.thread_pool import thread_pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests

from core.domain.screening.exclusion import is_excluded
from core.data.sources.tencent_kline import fetch_tencent_kline_raw
from core.data.sources.qfq_normalizer import prepare_qfq_daily_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fast_collect")


def _latest_trading_day_str():
    try:
        from core.shared.trading_time import get_latest_trading_day
        return get_latest_trading_day().strftime("%Y-%m-%d")
    except Exception:
        return datetime.date.today().strftime("%Y-%m-%d")


def get_all_codes(pool="all"):
    """获取股票代码列表"""
    codes = []
    if pool in ("all", "csi_all"):
        try:
            import baostock as bs
            bs.login()
            rs = bs.query_all_stock(day=_latest_trading_day_str())
            while (rs.error_code == "0") and rs.next():
                row = rs.get_row_data()
                if len(row) >= 1:
                    raw = row[0]
                    if "." in raw:
                        raw = raw.split(".")[1]
                    if raw.isdigit() and len(raw) == 6:
                        codes.append(raw)
            bs.logout()
        except Exception:
            pass
    if pool in ("csi300", "csi_all"):
        try:
            from core.data.sources.tushare import get_index_components
            codes.extend(get_index_components("000300"))
        except Exception:
            pass
    if pool in ("csi500", "csi_all"):
        try:
            from core.data.sources.tushare import get_index_components
            codes.extend(get_index_components("000905"))
        except Exception:
            pass
    if not codes:
        try:
            from core.data.sources.tushare import get_all_stock_codes
            codes = get_all_stock_codes()
            if not codes:
                raise Exception("无法获取股票列表")
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise
    return [c for c in list(set(codes))
            if not is_excluded(c)[0]]


# ═══════════════════ TuShare (前复权价 + adj_factor) ═══════════════════

def fetch_one_tushare(code, start_date="20120101", end_date=None):
    """从 TuShare 获取前复权价 + 后复权因子，返回 (code, df)。
    df 包含前复权 OHLC、股数口径 volume、元口径 amount 和后复权因子 adj_factor。
    失败返回 None。
    """
    try:
        from core.data.sources.tushare import get_daily_kline, get_adj_factor
        if end_date is None:
            from core.shared.trading_time import get_latest_trading_day
            end_date = get_latest_trading_day().strftime("%Y%m%d")

        price_df = None
        for attempt in range(3):
            price_df = get_daily_kline(code, start_date=start_date, end_date=end_date, adj="qfq")
            if price_df is not None and not price_df.empty:
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
        if price_df is None or price_df.empty:
            return None

        adj_df = None
        for attempt in range(3):
            adj_df = get_adj_factor(code, start_date=start_date, end_date=end_date)
            if adj_df is not None and not adj_df.empty:
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
        if adj_df is None or adj_df.empty:
            return None
        adj_df = adj_df.rename(columns={"trade_date": "date"})
        price_df["date"] = pd.to_datetime(price_df["date"])
        adj_df["date"] = pd.to_datetime(adj_df["date"])
        price_df = price_df.merge(adj_df, on="date", how="left")
        if price_df["adj_factor"].isna().any():
            return None

        price_df["amplitude"] = price_df.get("amplitude")
        price_df["pct_chg"] = price_df.get("pct_chg")
        price_df["turnover"] = price_df.get("turnover")
        price_df.attrs["quote_source"] = "tushare"
        return (code, price_df)
    except Exception:
        return None


# ═══════════════════ 腾讯 IFZQ ═══════════════════

def fetch_one_tencent(code, timeout=5, return_error=False):
    """从腾讯 IFZQ 获取一只股票的未复权日线（~640天）"""
    try:
        code, rows = fetch_tencent_kline_raw(code, adj="", timeout=timeout)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.sort_values("date").reset_index(drop=True)
        df["pct_chg"] = df["close"].astype(float).pct_change(fill_method=None).fillna(0) * 100
        df["amplitude"] = None
        df["turnover"] = None
        df.attrs["quote_source"] = "tencent"
        return (code, df)
    except Exception as error:
        logger.warning("[Tencent] 拉取或解析失败 %s: %s", code, error)
        if return_error:
            return {"code": code, "stage": "fetch", "error": str(error)}
        return None


def save_batch(code, df, source=None):
    """单条写入 MySQL（线程安全）"""
    from core.data import DailyQuotesRepo
    DailyQuotesRepo.save_batch(code, prepare_qfq_daily_frame(code, df, source=source or df.attrs.get("quote_source")))


def collect_tushare_snapshots(codes, trade_dates):
    from core.data.sources.tushare import get_pro, _rate_limit

    pro = get_pro()
    if not pro:
        return 0

    allowed = set(codes)
    frames = []
    for trade_date in trade_dates:
        ymd = trade_date.strftime("%Y%m%d")
        try:
            _rate_limit()
            price = pro.daily(trade_date=ymd)
            _rate_limit()
            factor = pro.adj_factor(trade_date=ymd)
            if price is None or price.empty or factor is None or factor.empty:
                logger.warning(f"[TuShare] {ymd} 批量日线或复权因子为空")
                continue
            price = price.copy()
            factor = factor[["ts_code", "trade_date", "adj_factor"]].copy()
            frame = price.merge(factor, on=["ts_code", "trade_date"], how="left")
            frame["code"] = frame["ts_code"].astype(str).str.split(".").str[0]
            frame = frame[frame["code"].isin(allowed)]
            if not frame.empty:
                frames.append(frame)
                logger.info(f"[TuShare] {ymd} 批量获取: {len(frame)} 只")
        except Exception as e:
            logger.warning(f"[TuShare] {ymd} 批量获取失败: {e}")

    if not frames:
        return 0

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["trade_date"], format="%Y%m%d")
    data = data.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    data["adj_factor"] = pd.to_numeric(data["adj_factor"], errors="coerce")
    data["adj_factor"] = data.groupby("code")["adj_factor"].transform(lambda s: s.ffill().bfill())
    data = data.dropna(subset=["adj_factor"])
    if data.empty:
        return 0

    anchors = data.groupby("code")["adj_factor"].last().dropna().to_dict()
    for col in ["open", "high", "low", "close"]:
        raw = pd.to_numeric(data[col], errors="coerce")
        anchor = data["code"].map(anchors).astype(float)
        data[col] = (raw * data["adj_factor"] / anchor).round(4)
    data["volume"] = pd.to_numeric(data["vol"], errors="coerce") * 100
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce") * 1000
    calculated_pct = data.groupby("code")["close"].pct_change(fill_method=None) * 100
    data["pct_chg"] = calculated_pct.fillna(pd.to_numeric(data.get("pct_chg"), errors="coerce")).fillna(0)
    data["amplitude"] = None
    data["turnover"] = None

    saved = 0
    columns = [
        "date", "open", "high", "low", "close", "volume", "amount",
        "amplitude", "pct_chg", "turnover", "adj_factor",
    ]
    for code, frame in data.groupby("code"):
        try:
            frame.attrs["quote_source"] = "tushare"
            save_batch(code, frame[columns].reset_index(drop=True), source="tushare")
            saved += 1
        except Exception as e:
            logger.warning(f"[TuShare] 批量日线写入失败 {code}: {e}")
    logger.info(f"[TuShare] 批量增量完成: 成功写入 {saved}/{data['code'].nunique()} 只")
    return saved


# ═══════════════════ BaoStock ═══════════════════

def _baostock_query(bs, code, start_date, end_date):
    """BaoStock 单次查询，返回 (code, df) 或 None"""
    mkt = "sh" if code.startswith(("6", "5")) else "sz"
    full_code = f"{mkt}.{code}"
    try:
        rs = bs.query_history_k_data_plus(
            full_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            return None
        rows = []
        while rs.next():
            row = rs.get_row_data()
            try:
                if row[0] == "":
                    continue
                rows.append({
                    "date": row[0].replace("-", ""),
                    "open": float(row[1]) if row[1] else None,
                    "high": float(row[2]) if row[2] else None,
                    "low": float(row[3]) if row[3] else None,
                    "close": float(row[4]) if row[4] else None,
                    "volume": float(row[5]) if row[5] else None,
                    "amount": float(row[6]) if row[6] else None,
                    "turnover": float(row[7]) if row[7] else None,
                    "pct_chg": float(row[8]) if row[8] else None,
                })
            except (IndexError, ValueError, TypeError):
                continue
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["amplitude"] = None
        df.attrs["quote_source"] = "baostock"
        return (code, df)
    except Exception:
        return None


def _prepare_baostock_qfq(code, df, start_date, end_date):
    from core.data.sources.tushare import get_adj_factor

    factors = get_adj_factor(
        code,
        start_date=str(start_date).replace("-", ""),
        end_date=str(end_date).replace("-", ""),
    )
    if factors is None or factors.empty:
        return None
    factors = factors.copy()
    factor_date = "date" if "date" in factors.columns else "trade_date"
    factors["date"] = pd.to_datetime(factors[factor_date])
    factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared = prepared.merge(factors[["date", "adj_factor"]], on="date", how="left")
    prepared["adj_factor"] = prepared["adj_factor"].ffill().bfill()
    if prepared["adj_factor"].isna().any():
        return None
    new_anchor = float(prepared["adj_factor"].iloc[-1])
    if new_anchor <= 0:
        return None
    for col in ("open", "high", "low", "close"):
        prepared[col] = (
            pd.to_numeric(prepared[col], errors="coerce")
            * prepared["adj_factor"] / new_anchor
        ).round(4)
    calculated = prepared["close"].pct_change(fill_method=None) * 100
    prepared["pct_chg"] = calculated.fillna(
        pd.to_numeric(prepared.get("pct_chg"), errors="coerce")
    ).fillna(0)
    prepared.attrs["quote_source"] = "baostock"
    return prepared


def _baostock_with_fallback(bs, code, start_date, end_date, retries=3):
    """BaoStock 查询 + decode 失败重试，仍失败回退 TuShare"""
    for attempt in range(retries):
        result = _baostock_query(bs, code, start_date, end_date)
        if result is not None:
            prepared = _prepare_baostock_qfq(code, result[1], start_date, end_date)
            if prepared is not None:
                return code, prepared
            break
        if attempt < retries - 1:
            time.sleep(1.0)
    result = fetch_one_tushare(code)
    if result is None:
        return None
    _, df = result
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if df.empty:
        return None
    return (code, df)


def fetch_one_baostock(bs, code):
    """从 BaoStock 获取一只股票的完整日线数据"""
    return _baostock_with_fallback(bs, code, "2012-01-01", _latest_trading_day_str())


def fetch_one_baostock_range(bs, code, start_date, end_date):
    """从 BaoStock 获取指定日期范围的日线数据"""
    return _baostock_with_fallback(bs, code, start_date, end_date)


# ═══════════════════ 主流程 ═══════════════════

def run_tushare(codes, args):
    """TuShare 前复权价 + adj_factor 并行采集"""
    lock = threading.Lock()
    success = fail = saved = 0
    total = len(codes)

    logger.info(f"[TuShare] 原始价采集 ({args.workers} 线程)...")
    t0 = time.time()

    with thread_pool.default.borrow() as pool:
        futures = {pool.submit(fetch_one_tushare, c, "20120101"): c for c in codes}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result is not None:
                code, df = result
                try:
                    save_batch(code, df, source="tushare")
                    with lock:
                        saved += 1
                        success += 1
                except Exception:
                    with lock:
                        fail += 1
            else:
                with lock:
                    fail += 1

            done = i + 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                with lock:
                    sc, fl, sv = success, fail, saved
                logger.info(f"[TuShare] {done}/{total} ({rate:.0f}只/s), 成功 {sc}, 失败 {fl}, 写入 {sv}")

    elapsed = time.time() - t0
    logger.info(f"[TuShare] 完成: {elapsed:.1f}s, 成功 {success}, 失败 {fail}")
    return saved


def run_tencent(codes, args, return_details=False):
    """腾讯 IFZQ 并行采集"""
    lock = threading.Lock()
    success = fail = saved = 0
    total = len(codes)
    successful_codes = []
    failed_codes = []
    actual_start = None
    actual_end = None

    logger.info(f"开始采集 ({args.workers} 线程)...")
    t0 = time.time()

    with thread_pool.default.borrow() as pool:
        futures = {pool.submit(fetch_one_tencent, c, args.timeout, return_error=return_details): c for c in codes}
        for i, fut in enumerate(as_completed(futures)):
            code = futures[fut]
            fetch_failed = False
            try:
                result = fut.result()
            except Exception as error:
                result = None
                fetch_failed = True
                logger.warning("[Tencent] 拉取失败 %s: %s", code, error)
                failed_codes.append({"code": code, "stage": "fetch", "error": str(error)})
            if isinstance(result, dict) and result.get("stage") == "fetch":
                failed_codes.append(result)
                fetch_failed = True
                result = None
            if result is not None:
                code, df = result
                try:
                    save_batch(code, df, source="tencent")
                    successful_codes.append(code)
                    if not df.empty and "date" in df.columns:
                        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
                        if not dates.empty:
                            first_date = dates.min()
                            last_date = dates.max()
                            actual_start = first_date if actual_start is None else min(actual_start, first_date)
                            actual_end = last_date if actual_end is None else max(actual_end, last_date)
                    with lock:
                        saved += 1
                        success += 1
                except Exception as error:
                    logger.warning("[Tencent] 写入失败 %s: %s", code, error)
                    failed_codes.append({"code": code, "stage": "write", "error": str(error)})
                    with lock:
                        fail += 1
            else:
                if not fetch_failed:
                    failed_codes.append({"code": code, "stage": "fetch", "error": "no_data_or_fetch_failed"})
                with lock:
                    fail += 1

            done = i + 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                with lock:
                    sc, fl, sv = success, fail, saved
                logger.info(f"进度: {done}/{total} ({rate:.0f}只/s), 成功 {sc}, 失败 {fl}, 已写入 {sv}")

    elapsed = time.time() - t0
    logger.info(f"采集完成: {elapsed:.1f}s, 成功 {success}, 失败 {fail}")
    if return_details:
        date_range = ({
            "start": pd.Timestamp(actual_start).date().isoformat(),
            "end": pd.Timestamp(actual_end).date().isoformat(),
        } if actual_start is not None else None)
        return {
            "expected": total,
            "success": success,
            "failed": fail,
            "skipped": 0,
            "successful_codes": successful_codes,
            "failed_codes": failed_codes,
            "actual_date_range": date_range,
        }
    return saved


def run_baostock(codes, args):
    """BaoStock 串行采集"""
    import baostock as bs
    bs.login()
    total = len(codes)
    success = fail = saved = 0

    logger.info(f"开始采集 (串行, 间隔 {args.delay}s)...")
    t0 = time.time()

    for i, code in enumerate(codes):
        try:
            result = fetch_one_baostock(bs, code)
            if result is not None:
                code, df = result
                try:
                    save_batch(code, df)
                    saved += 1
                    success += 1
                except Exception:
                    fail += 1
            else:
                fail += 1
        except Exception:
            fail += 1

        done = i + 1
        if done % 100 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            logger.info(
                f"进度: {done}/{total} ({rate:.1f}只/s), "
                f"成功 {success}, 失败 {fail}, 已写入 {saved}, "
                f"预计剩余 {remaining:.0f}s"
            )

        if args.delay > 0:
            time.sleep(args.delay)

    bs.logout()
    elapsed = time.time() - t0
    logger.info(f"采集完成: {elapsed:.1f}s, 成功 {success}, 失败 {fail}")
    return saved


def run_hybrid(codes, args):
    """混合模式: 腾讯快速填充 → BaoStock 覆盖残缺数据 + 补全历史"""
    import baostock as bs
    from core.data import Database

    # Phase 1: 腾讯并行 (~3 分钟) — 快速拿到各股票数据骨架
    logger.info("=" * 50)
    logger.info("Phase 1/3: 腾讯 IFZQ 并行采集 (~3 min)")
    logger.info("=" * 50)
    saved = run_tencent(codes, args)

    # Phase 2: 收集需要补全的股票
    logger.info("=" * 50)
    logger.info("Phase 2/3: 分析数据完整性")
    conn = Database.get_connection()
    try:
        cur = conn.cursor()

        # 找到 amount=0 的股票（腾讯数据残缺）及其日期范围
        cur.execute(
            "SELECT code, MIN(trade_date) as zero_start, MAX(trade_date) as zero_end, "
            "COUNT(*) as zero_cnt FROM daily_quotes "
            "WHERE amount = 0 OR amount IS NULL GROUP BY code"
        )
        zero_amount = {r["code"]: (str(r["zero_start"])[:10], str(r["zero_end"])[:10], r["zero_cnt"])
                       for r in cur.fetchall()}

        # 找到数据量不足的股票（可能只有腾讯数据，缺历史）
        cur.execute(
            "SELECT code, MIN(trade_date) as earliest, COUNT(*) as cnt "
            "FROM daily_quotes GROUP BY code"
        )
        db_stats = {r["code"]: (str(r["earliest"])[:10], r["cnt"]) for r in cur.fetchall()}
        cur.close()
    finally:
        Database.return_connection(conn)

    full_backfill = []   # 全量 2012-2026 (没有 BaoStock 数据的股票)
    partial_backfill = []  # 只覆盖腾讯残缺段

    for code in codes:
        if code not in db_stats:
            full_backfill.append(code)
            continue

        earliest, cnt = db_stats[code]
        has_amount_zero = code in zero_amount

        if cnt >= 2000 and not has_amount_zero:
            continue  # 数据完整，跳过

        if not has_amount_zero:
            # 只有少量数据但不是 amount=0（罕见，可能是纯 BaoStock 只采了部分）
            if earliest > "2012-01-02":
                end_date = (pd.to_datetime(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                full_backfill.append((code, "2012-01-01", end_date))
            else:
                full_backfill.append(code)
            continue

        zero_start, zero_end, zero_cnt = zero_amount[code]

        if cnt < 200:
            # 几乎只有腾讯数据，全量 BaoStock
            full_backfill.append(code)
        else:
            # 有混合数据：BaoStock 覆盖腾讯的残缺段
            # 从 amount=0 的开始前推几天做缓冲区
            buf_start = (pd.to_datetime(zero_start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            partial_backfill.append((code, buf_start, zero_end))

    total_tasks = len(full_backfill) + len(partial_backfill)
    if total_tasks == 0:
        logger.info("数据已完整，无需补全")
        return saved

    logger.info(
        f"需补全 {total_tasks} 只: "
        f"全量 {len(full_backfill)} 只, "
        f"覆盖残缺段 {len(partial_backfill)} 只"
    )

    # Phase 3: BaoStock 补全
    logger.info("=" * 50)
    logger.info("Phase 3/3: BaoStock 补全 (覆盖 amount=0 + 补历史)")
    logger.info("=" * 50)

    bs.login()
    success = fail = 0
    t0 = time.time()
    all_tasks = []

    for code in full_backfill:
        if isinstance(code, tuple):
            all_tasks.append(code)
        else:
            all_tasks.append((code, "2012-01-01", _latest_trading_day_str()))
    all_tasks.extend(partial_backfill)

    for i, (code, start, end) in enumerate(all_tasks):
        try:
            result = fetch_one_baostock_range(bs, code, start, end)
            if result is not None:
                _, df = result
                try:
                    save_batch(code, df)
                    success += 1
                    saved += 1
                except Exception:
                    fail += 1
            else:
                fail += 1
        except Exception:
            fail += 1

        done = i + 1
        if done % 100 == 0 or done == total_tasks:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total_tasks - done) / rate if rate > 0 else 0
            logger.info(
                f"补全进度: {done}/{total_tasks} ({rate:.1f}只/s), "
                f"成功 {success}, 失败 {fail}, 预计剩余 {remaining:.0f}s"
            )

        if args.delay > 0:
            time.sleep(args.delay)

    bs.logout()
    elapsed = time.time() - t0
    logger.info(f"补全完成: {elapsed:.1f}s, 成功 {success}, 失败 {fail}")
    return saved


def detect_gaps(lookback_days=5):
    """检测近期有数据缺口的股票。返回 {code: [missing_date_str, ...]}"""
    from core.data import Database
    from core.shared.trading_time import is_trading_day, get_latest_trading_day
    import pandas as pd

    conn = Database.get_connection()
    try:
        cursor = conn.cursor()

        today = get_latest_trading_day()
        yesterday = today - datetime.timedelta(days=1)
        dates_in_window = []
        d = today - datetime.timedelta(days=lookback_days)
        while d <= yesterday:
            if is_trading_day(d):
                dates_in_window.append(d.strftime("%Y-%m-%d"))
            d += datetime.timedelta(days=1)

        if not dates_in_window:
            return {}

        cursor.execute(
            "SELECT code, trade_date FROM daily_quotes "
            "WHERE trade_date >= %s ORDER BY code, trade_date",
            (dates_in_window[0],),
        )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        Database.return_connection(conn)

    stock_dates = {}
    for r in rows:
        code = r["code"]
        dt_str = r["trade_date"].strftime("%Y-%m-%d") if hasattr(r["trade_date"], "strftime") else str(r["trade_date"])[:10]
        stock_dates.setdefault(code, set()).add(dt_str)

    gaps = {}
    for code, have_dates in stock_dates.items():
        # ETF/基金 跳过（交易特性不同，常有停牌）
        if is_excluded(code)[0]:
            continue
        # 新股跳过：最早交易日如果在窗口内，说明上市不足窗口天数
        first_date = min(have_dates) if have_dates else None
        if first_date and first_date > dates_in_window[0]:
            continue
        missing = [dt for dt in dates_in_window if dt not in have_dates]
        if missing:
            gaps[code] = missing

    return gaps


def collect_incremental(codes, source="tencent", threads=32, delay=0.1, lookback_days=3, latest_trading_day=None, return_details=False):
    """增量采集: 检测 MAX(trade_date) + 近期缺口，只拉取有缺失的股票"""
    from core.data import Database
    from core.shared.trading_time import is_trading_day, get_latest_trading_day
    import pandas as pd

    if isinstance(latest_trading_day, str):
        latest_trading_day = datetime.date.fromisoformat(latest_trading_day[:10])
    if latest_trading_day is None:
        latest_trading_day = get_latest_trading_day()
    today = latest_trading_day  # 统一用最近交易日, 不再用自然日
    today_str = today.strftime("%Y-%m-%d")

    conn = Database.get_connection()
    try:
        cursor = conn.cursor()

        # 1. 查每个股票的最新日期 + 近 N 天记录条数
        lookback_start = (today - datetime.timedelta(days=lookback_days + 2)).strftime("%Y-%m-%d")
        cursor.execute(
            "SELECT code, MAX(trade_date) as last_date, "
            "SUM(CASE WHEN trade_date >= %s THEN 1 ELSE 0 END) as cnt_recent "
            "FROM daily_quotes WHERE code IS NOT NULL GROUP BY code",
            (lookback_start,),
        )
        existing = {}
        for r in cursor.fetchall():
            existing[r["code"]] = (r["last_date"], r["cnt_recent"] or 0)
        cursor.close()
    finally:
        Database.return_connection(conn)

    # 2. 计算窗口内期望的交易日数
    expected_days = 0
    d = today - datetime.timedelta(days=lookback_days + 2)
    while d <= today:
        if is_trading_day(d):
            expected_days += 1
        d += datetime.timedelta(days=1)

    if source == "tushare":
        need_codes = []
        for c in codes:
            info = existing.get(c)
            if info is None or info[0] < today:
                need_codes.append(c)
        skipped = len(codes) - len(need_codes)
        logger.info(f"TuShare 增量: {len(need_codes)}/{len(codes)} 需要更新, 跳过 {skipped} 只已更新股票")
        if not need_codes:
            logger.info("数据已是最新，无需采集")
            return 0
        trade_dates = []
        d = today - datetime.timedelta(days=lookback_days + 2)
        while d <= today:
            if is_trading_day(d):
                trade_dates.append(d)
            d += datetime.timedelta(days=1)
        return collect_tushare_snapshots(codes, trade_dates)

    elif source == "tencent":
        need_codes = []
        gap_codes = []
        for c in codes:
            info = existing.get(c)
            if info is None:
                need_codes.append(c)
            else:
                last_date, cnt_recent = info
                # 最新日期不是今天 → 需要更新
                if last_date < today:
                    need_codes.append(c)
                # 最新日期是今天但近期有缺口 → 也需要更新
                elif cnt_recent < expected_days:
                    need_codes.append(c)
                    gap_codes.append(c)

        skipped = len(codes) - len(need_codes)
        if gap_codes:
            logger.info(f"检测到 {len(gap_codes)} 只有数据缺口 (有今日但缺中间日期): {gap_codes[:10]}...")
        logger.info(f"Tencent 增量: {len(need_codes)}/{len(codes)} 需要更新, 跳过 {skipped} 只完整")
        if not need_codes:
            logger.info("数据已是最新且完整，无需采集")
            if return_details:
                return {
                    "expected": len(codes), "success": 0, "failed": 0, "skipped": skipped,
                    "successful_codes": [], "failed_codes": [], "actual_date_range": None,
                }
            return 0
        ns = argparse.Namespace(workers=threads, timeout=5)
        result = run_tencent(need_codes, ns, return_details=return_details)
        if return_details:
            result["expected"] = len(codes)
            result["skipped"] = skipped
            return result
        return result

    elif source == "baostock":
        import baostock as bs
        need_codes = []
        for c in codes:
            info = existing.get(c)
            if info is None or info[0] is None or info[0] < today:
                need_codes.append(c)
        skipped = len(codes) - len(need_codes)
        logger.info(f"BaoStock 增量: {len(need_codes)}/{len(codes)} 需要更新, 跳过 {skipped} 只已更新股票")
        if not need_codes:
            logger.info("数据已是最新，无需采集")
            return 0

        bs.login()
        saved = 0
        total = len(need_codes)
        t0 = time.time()
        today_str = today.strftime("%Y-%m-%d")

        for i, code in enumerate(need_codes):
            info = existing.get(code)
            if info is None:
                start = "2012-01-01"
            else:
                last_date, cnt_recent = info
                start_dt = pd.Timestamp(last_date) - pd.Timedelta(days=lookback_days)
                start = start_dt.strftime("%Y-%m-%d")

            try:
                result = fetch_one_baostock_range(bs, code, start, today_str)
                if result is not None:
                    _, df = result
                    save_batch(code, df)
                    saved += 1
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / rate if rate > 0 else 0
                logger.info(f"增量进度: {i+1}/{total} ({rate:.1f}只/s), 已写入 {saved}, 预计剩余 {remaining:.0f}s")

            if delay > 0:
                time.sleep(delay)

        bs.logout()
        return saved

    else:
        raise ValueError(f"不支持的增量采集数据源: {source}")


def main():
    parser = argparse.ArgumentParser(description="服务器极速数据采集")
    parser.add_argument("--source", default="tushare", choices=["tushare", "tencent", "baostock", "hybrid"],
                        help="tushare(推荐:qfq+复权因子) / tencent(快,~640天qfq) / baostock(完整) / hybrid(腾讯+补历史)")
    parser.add_argument("--workers", type=int, default=32, help="并发线程数 (仅 tencent)")
    parser.add_argument("--pool", default="all", choices=["all", "csi300", "csi500"])
    parser.add_argument("--timeout", type=int, default=5, help="HTTP 超时 (仅 tencent)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="BaoStock 请求间隔秒数 (默认 0.1)")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--retry-missing", action="store_true", help="只补采<100条的股票")
    parser.add_argument("--check-gaps", type=int, default=0, metavar="DAYS",
                        help="检测近 N 天的数据缺口 (不采集, 仅报告)")
    args = parser.parse_args()

    if args.check_gaps > 0:
        gaps = detect_gaps(args.check_gaps)
        if not gaps:
            logger.info(f"近 {args.check_gaps} 天数据完整，未发现缺口")
        else:
            gap_list = sorted(gaps.items(), key=lambda x: -len(x[1]))
            logger.info(f"发现 {len(gaps)} 只有数据缺口 (近{args.check_gaps}天):")
            for code, missing in gap_list[:30]:
                logger.info(f"  {code}: 缺 {missing}")
            if len(gap_list) > 30:
                logger.info(f"  ... 还有 {len(gap_list)-30} 只")
        return

    codes = get_all_codes(args.pool)
    total = len(codes)
    logger.info(f"数据源: {args.source}, 股票池: {args.pool}, 共 {total} 只")

    if args.retry_missing:
        from core.data import Database
        conn = Database.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT code FROM daily_quotes GROUP BY code HAVING COUNT(*) < 100")
            missing = {r["code"] for r in cur.fetchall()}
            cur.close()
        finally:
            Database.return_connection(conn)
        codes = [c for c in codes if c in missing]
        logger.info(f"补采模式: {len(codes)} 只缺数据股票")

    if args.source == "tushare":
        saved = run_tushare(codes, args)
    elif args.source == "hybrid":
        saved = run_hybrid(codes, args)
    elif args.source == "baostock":
        saved = run_baostock(codes, args)
    else:
        saved = run_tencent(codes, args)

    if not args.no_sync and saved > 0:
        logger.info("增量同步到 qlib...")
        from core.shared import config
        from core.adapters.qlib.bridge import QlibDataBridge
        bridge = QlibDataBridge(config.QLIB_DATA_DIR)
        result = bridge.sync_incremental()
        logger.info(f"qlib 同步完成: {result}")


if __name__ == "__main__":
    main()
