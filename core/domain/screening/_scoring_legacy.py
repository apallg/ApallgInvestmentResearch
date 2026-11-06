"""
L0 因子评分函数 + 数据采集函数 + 冲击成本估计
从 buy_side.py 提取, 方便 AI 按因子维度独立理解和修改
"""
import datetime
import logging
import math as _math
import re
import threading
import time
from concurrent.futures import as_completed
from core.shared.infrastructure.thread_pool import thread_pool

import numpy as np
import pandas as pd

from core.shared import config
from core.shared.persistence import today_date

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 冲击成本估计 (Almgren-Chriss 简化模型)
# ══════════════════════════════════════════════

def estimate_impact_cost(close, volume, amount=None, daily_volume=None,
                         trade_amount=None, eta=None):
    """线性冲击模型: impact = eta * sigma_daily * (participation_rate)^0.5

    - 用最近20日波动率做 sigma_daily 代理
    - 参与率 = trade_amount / daily_volume
    - eta 经验常数, A股 ~0.15

    返回冲击成本比例, e.g. 0.005 表示 0.5% (50bp)
    无法估计时返回 0.0
    """
    try:
        eta = eta if eta is not None else getattr(config, "BUY_SIDE_IMPACT_ETA", 0.15)
        close = float(close)
        vol = float(volume)
        if close <= 0 or vol <= 0:
            return 0.0
        dv = daily_volume if daily_volume else vol
        if dv <= 0:
            return 0.0
        ta = trade_amount if trade_amount else (dv * close * 0.01)
        participation = ta / (dv * close)
        if participation <= 0:
            return 0.0
        volatility = _estimate_daily_volatility(close, amount)
        impact = eta * max(volatility, 0.005) * (participation ** 0.5)
        return round(min(impact, 0.05), 6)
    except Exception:
        return 0.0


def _estimate_daily_volatility(close, amount=None):
    """用收盘价历史序列估算日波动率, 无法估计时返回 0.02 (中位数默认值)"""
    try:
        if amount is not None and hasattr(amount, 'pct_change'):
            rets = amount.pct_change(fill_method=None).dropna()
            if len(rets) >= 5:
                return max(float(rets.std()), 0.005)
        return 0.02
    except Exception:
        return 0.02


def apply_impact_penalty(score, impact_cost, lambda_impact=None):
    """从 Alpha 分中预扣冲击成本: net_score = score - lambda * tanh(impact * 20)

    tanh 压缩防止极端冲击成本过度惩罚, lambda 控制整体惩罚力度
    """
    try:
        lb = lambda_impact if lambda_impact is not None else getattr(
            config, "BUY_SIDE_IMPACT_LAMBDA", 0.3)
        if lb <= 0 or impact_cost <= 0:
            return score
        penalty = lb * _math.tanh(impact_cost * 20)
        return round(score - penalty, 4)
    except Exception:
        return score


# ══════════════════════════════════════════════
# L0 因子评分函数 (每只返回 -1..1)
# ══════════════════════════════════════════════

def _score_momentum(df, n=20, close_arr=None):
    """N日动量 + 相对强弱 → -1..1"""
    try:
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        ret_n = (closes.iloc[-1] / closes.iloc[-n] - 1) if closes.iloc[-n] > 0 else 0
        half = closes.iloc[-n//2:] if n >= 10 else closes.iloc[-5:]
        slope_recent = (half.iloc[-1] - half.iloc[0]) / max(abs(half.iloc[0]), 0.01)
        raw = ret_n * 0.6 + slope_recent * 0.4
        return round(_math.tanh(raw * 2), 4)
    except Exception:
        return 0.0


def _score_volatility(df, n=20, close_arr=None, ret_arr=None):
    """低波动加分，高波动减分 → -1..1 (反向：波动越低分数越高)"""
    try:
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        returns = ret_arr if ret_arr is not None else closes.pct_change(fill_method=None)
        returns = returns.dropna().iloc[-n:]
        if len(returns) < 5:
            return 0.0
        vol = returns.std()
        mean_ret = returns.mean()
        sharpe_like = mean_ret / max(vol, 0.001)
        ann_vol = vol * (250 ** 0.5)
        vol_score = 1.0 - ann_vol / 0.6
        raw = sharpe_like * 0.3 + vol_score * 0.7
        return round(_math.tanh(raw * 1.5), 4)
    except Exception:
        return 0.0


def _compute_ht_score(df, ht_key=None):
    """华泰多因子评分: ht_key指定单因子,None则综合评分"""
    from core.domain.factors.huatai import compute_single_factor, compute_huatai_score
    if ht_key:
        return compute_single_factor(df, ht_key)
    return compute_huatai_score(df)


# ═══ 基本面 ═══

def _prefetch_financials(codes, max_workers=12, progress_cb=None):
    """预取全部股票的完整财务历史 {code: [(report_date, {指标: 值}), ...]}，report_date 升序"""
    from core.data import Database
    cache = {}
    total = len(codes)

    if Database.is_available():
        logger.info(f"[财务预取] 从 MySQL 读取 {total} 只...")
        placeholders = ",".join(["%s"] * len(codes))
        rows = Database.fetchall(
            f"SELECT f.code, f.report_date, GREATEST(f.ann_date, DATE(v.source_observed_at)) AS avail_date, "
            f"f.roe, f.net_profit_yoy, f.revenue_yoy, f.debt_to_assets "
            f"FROM fundamental_data f INNER JOIN ("
            f"SELECT code, report_date, ann_date, MAX(source_observed_at) AS source_observed_at "
            f"FROM fundamental_data_versions WHERE source_name='tushare_fina_indicator' "
            f"AND version_kind='provider_observation' AND ann_date IS NOT NULL "
            f"AND code IN ({placeholders}) "
            f"GROUP BY code, report_date, ann_date"
            f") v ON v.code=f.code AND v.report_date=f.report_date AND v.ann_date=f.ann_date "
            f"WHERE f.code IN ({placeholders}) AND f.ann_date IS NOT NULL "
            f"ORDER BY f.code, f.report_date",
            tuple(codes) + tuple(codes),
        )
        if rows:
            for r in rows:
                code = r["code"]
                rd = r["report_date"]
                if hasattr(rd, "date"):
                    rd = rd.date()
                elif isinstance(rd, str):
                    rd = datetime.date.fromisoformat(rd)
                metrics = {}
                if r.get("roe") is not None:
                    metrics["净资产收益率"] = float(r["roe"])
                if r.get("net_profit_yoy") is not None:
                    metrics["净利润增长率"] = float(r["net_profit_yoy"])
                if r.get("revenue_yoy") is not None:
                    metrics["营业收入增长率"] = float(r["revenue_yoy"])
                if r.get("debt_to_assets") is not None:
                    metrics["资产负债率"] = float(r["debt_to_assets"])
                if metrics:
                    if code not in cache:
                        cache[code] = []
                    avail_date = r["avail_date"]
                    if hasattr(avail_date, "date"):
                        avail_date = avail_date.date()
                    elif isinstance(avail_date, str):
                        avail_date = datetime.date.fromisoformat(avail_date[:10])
                    cache[code].append((rd, avail_date, metrics))
            mysql_hit = len(cache)
            logger.info(f"[财务预取] MySQL命中: {mysql_hit}/{total} 只")
            if progress_cb:
                progress_cb(mysql_hit, total)
            if mysql_hit > total * 0.8:
                return cache

    logger.info(f"[财务预取] 完成: PIT 公告日有效 {len(cache)}/{total} 只；缺公告日数据不参与历史评分")
    return cache


def _score_fundamentals(code, fetcher=None, as_of=None, financial_cache=None):
    """基本面评分：ROE + 利润增速 + 营收增速 + 负债率 → -1..1"""
    try:
        fin = None
        if financial_cache and as_of is not None:
            periods = financial_cache.get(code, [])
            best = None
            for report_date, avail_date, metrics in periods:
                if avail_date is not None and avail_date <= as_of:
                    if best is None or report_date > best[0]:
                        best = (report_date, metrics)
            if best:
                fin = best[1]
            else:
                return 0.0
        else:
            if as_of is not None:
                try:
                    from core.data import Database
                    rows = Database.fetchall(
                        "SELECT f.report_date, GREATEST(f.ann_date, DATE(v.source_observed_at)) AS avail_date, "
                        "f.roe, f.net_profit_yoy, f.revenue_yoy, f.debt_to_assets "
                        "FROM fundamental_data f INNER JOIN ("
                        "SELECT code, report_date, ann_date, MAX(source_observed_at) AS source_observed_at "
                        "FROM fundamental_data_versions WHERE code=%s "
                        "AND source_name='tushare_fina_indicator' AND version_kind='provider_observation' "
                        "AND ann_date IS NOT NULL GROUP BY code, report_date, ann_date"
                        ") v ON v.code=f.code AND v.report_date=f.report_date AND v.ann_date=f.ann_date "
                        "WHERE f.code=%s AND f.ann_date IS NOT NULL "
                        "AND GREATEST(f.ann_date, DATE(v.source_observed_at)) <= %s "
                        "AND f.report_date >= DATE_SUB(%s, INTERVAL 2 YEAR) "
                        "ORDER BY f.report_date DESC LIMIT 1",
                        (code, code, as_of, as_of),
                    )
                    if rows:
                        r = rows[0]
                        fin = {}
                        if r.get("roe") is not None:
                            fin["净资产收益率"] = float(r["roe"])
                        if r.get("net_profit_yoy") is not None:
                            fin["净利润增长率"] = float(r["net_profit_yoy"])
                        if r.get("revenue_yoy") is not None:
                            fin["营业收入增长率"] = float(r["revenue_yoy"])
                        if r.get("debt_to_assets") is not None:
                            fin["资产负债率"] = float(r["debt_to_assets"])
                except Exception:
                    pass
            if not fin:
                return 0.0
        if not fin:
            return 0.0

        score = 0.0
        weight = 0.0

        roe_str = str(fin.get("净资产收益率", ""))
        roe = _parse_num(roe_str)
        if roe is not None:
            weight += 0.30
            score += _math.tanh(roe / 15.0) * 0.30

        pg_str = str(fin.get("净利润增长率", ""))
        pg = _parse_num(pg_str)
        if pg is not None:
            weight += 0.25
            score += _math.tanh(pg / 30.0) * 0.25

        rg_str = str(fin.get("营业收入增长率", ""))
        rg = _parse_num(rg_str)
        if rg is not None:
            weight += 0.15
            score += _math.tanh(rg / 20.0) * 0.15

        dr_str = str(fin.get("资产负债率", ""))
        dr = _parse_num(dr_str)
        if dr is not None:
            weight += 0.10
            score += _math.tanh((50.0 - dr) / 30.0) * 0.10

        pe_raw = fin.get("市盈率") or fin.get("PE") or fin.get("pe")
        if pe_raw is not None and pg is not None and pg > 0:
            pe_val = _parse_num(str(pe_raw))
            if pe_val is not None and pe_val > 0:
                peg = pe_val / pg
                weight += 0.10
                score += _math.tanh((2.0 - peg) / 2.0) * 0.10

        eps_raw = fin.get("基本每股收益") or fin.get("EPS") or fin.get("eps")
        if eps_raw is not None:
            eps_val = _parse_num(str(eps_raw))
            if eps_val is not None:
                weight += 0.10
                score += _math.tanh(eps_val / 2.0) * 0.10

        if weight < 0.2:
            return 0.0
        return round(score / weight, 4)
    except Exception:
        return 0.0


# ═══ 资金流向 ═══

def _score_capital_flow(code, fetcher=None, as_of=None, flow_cache=None):
    """资金流向评分：近期主力净流入占比 + 趋势 + 超大单/小单比 → -1..1"""
    try:
        from core.data import Database

        rows = None
        if flow_cache is not None:
            rows = flow_cache.get(code)
        elif as_of and Database.is_available():
            rows = Database.fetchall(
                "SELECT f.main_net_flow, f.trade_date, q.amount "
                "FROM capital_flow_daily f "
                "LEFT JOIN daily_quotes q ON q.code=f.code AND q.trade_date=f.trade_date "
                "WHERE f.code=%s AND f.trade_date <= %s "
                "ORDER BY f.trade_date DESC LIMIT 20",
                (code, as_of),
            )

        if rows:
            pcts = []
            for r in rows:
                amt = float(r.get("amount") or 0)
                if amt > 0:
                    pcts.append(float(r.get("main_net_flow") or 0) * 10000 / amt * 100)
            if pcts:
                avg_pct = sum(pcts[:5]) / min(len(pcts), 5)
                base = _math.tanh(avg_pct / 12.0)
                if len(pcts) >= 10:
                    avg5 = sum(pcts[:5]) / 5
                    avg20 = sum(pcts) / len(pcts)
                    if avg20 > 0 and avg5 > avg20 * 1.5:
                        base += 0.15
                    elif avg20 < 0 and avg5 < avg20 * 1.5:
                        base -= 0.15
                net_flows = [float(r.get("main_net_flow", 0) or 0) for r in rows[:5]]
                net_pos = sum(1 for f in net_flows if f > 0)
                if net_pos >= 4:
                    base += 0.15
                elif net_pos <= 1:
                    base -= 0.10
                return round(max(-1.0, min(1.0, base)), 3)
            total_net = sum(float(r.get("main_net_flow", 0) or 0) for r in rows[:5])
            return round(max(-1.0, min(1.0, total_net / 50000)), 3)
        if flow_cache is not None or as_of:
            return 0.0

        if fetcher is None:
            from core.data.fetcher import DataFetcher
            fetcher = DataFetcher()
        flow = fetcher.get_capital_flow(code)
        if flow is None or (isinstance(flow, pd.DataFrame) and flow.empty):
            return 0.0
        if isinstance(flow, pd.DataFrame):
            if as_of is not None:
                date_col = None
                for c in flow.columns:
                    if "日期" in str(c) or "时间" in str(c) or "date" in str(c).lower():
                        date_col = c
                        break
                if date_col:
                    flow[date_col] = pd.to_datetime(flow[date_col])
                    flow = flow[flow[date_col] <= pd.Timestamp(as_of)]
            recent = flow.head(10)
            net_col = None
            for c in recent.columns:
                if "净流入" in c or "主力" in c:
                    net_col = c
                    break
            if net_col is None:
                return 0.0
            values = pd.to_numeric(recent[net_col], errors="coerce").dropna()
            if len(values) < 2:
                return 0.0
            total_net = values.sum()
            avg_turnover = recent.get("成交额") or recent.get("成交金额") or 0
            if isinstance(avg_turnover, pd.Series) and len(avg_turnover) > 0:
                avg_turnover = pd.to_numeric(avg_turnover, errors="coerce").mean()
            else:
                avg_turnover = float(avg_turnover) if avg_turnover else 1e9
            if avg_turnover <= 0:
                avg_turnover = 1e9
            ratio = total_net / avg_turnover / 10
        else:
            return 0.0
        return round(max(-1.0, min(1.0, ratio * 5)), 3)
    except Exception:
        return 0.0


def fetch_and_save_capital_flow(codes=None, date=None, max_workers=2):
    """采资金流向数据到 MySQL。Tushare 优先, akshare 回退。"""
    from core.data import Database
    if not Database.is_available():
        logger.warning("[资金流向] MySQL 不可用")
        return 0

    Database.execute("""
        CREATE TABLE IF NOT EXISTS `capital_flow_daily` (
          id INT AUTO_INCREMENT PRIMARY KEY,
          code VARCHAR(10) NOT NULL,
          trade_date DATE NOT NULL,
          main_net_flow DECIMAL(16,2) DEFAULT 0,
          main_net_pct DECIMAL(10,4) DEFAULT 0,
          turnover DECIMAL(16,2) DEFAULT 0,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uk_code_date (code, trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    if date is None:
        date = today_date()
    trade_date_str = date.strftime("%Y%m%d")

    from core.data.sources.tushare import get_pro
    pro = get_pro()
    if pro:
        try:
            df = pro.moneyflow(trade_date=trade_date_str, limit=6000)
            if df is not None and not df.empty:
                params_list = []
                for _, row in df.iterrows():
                    try:
                        code = row["ts_code"].split(".")[0]
                        # moneyflow 金额单位: 万元; 净额 = 买入 - 卖出
                        buy_sm = float(row.get("buy_sm_amount", 0) or 0)
                        sell_sm = float(row.get("sell_sm_amount", 0) or 0)
                        buy_md = float(row.get("buy_md_amount", 0) or 0)
                        sell_md = float(row.get("sell_md_amount", 0) or 0)
                        buy_lg = float(row.get("buy_lg_amount", 0) or 0)
                        sell_lg = float(row.get("sell_lg_amount", 0) or 0)
                        buy_elg = float(row.get("buy_elg_amount", 0) or 0)
                        sell_elg = float(row.get("sell_elg_amount", 0) or 0)
                        net_amt = float(row.get("net_mf_amount", 0) or 0)
                        total_amount = buy_sm + sell_sm + buy_md + sell_md + buy_lg + sell_lg + buy_elg + sell_elg
                        main_pct = round(net_amt / total_amount * 100, 4) if total_amount > 0 else 0.0
                        params_list.append((code, date,
                            net_amt,
                            main_pct,
                            buy_sm - sell_sm, buy_md - sell_md, buy_lg - sell_lg, buy_elg - sell_elg))
                    except Exception:
                        pass
                saved = Database.executemany(
                    "INSERT INTO capital_flow_daily "
                    "(code, trade_date, main_net_flow, main_net_pct, "
                    "sm_net_flow, md_net_flow, lg_net_flow, elg_net_flow) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "main_net_flow=VALUES(main_net_flow), "
                    "main_net_pct=VALUES(main_net_pct), "
                    "sm_net_flow=VALUES(sm_net_flow), md_net_flow=VALUES(md_net_flow), "
                    "lg_net_flow=VALUES(lg_net_flow), elg_net_flow=VALUES(elg_net_flow)",
                    params_list)
                logger.info(f"[资金流向] Tushare {date} 采集完成: {saved} 只 (1次API调用)")
                return saved
            else:
                logger.info(f"[资金流向] Tushare {date} 无数据 (非交易日)")
                return 0
        except Exception as e:
            logger.error(f"[资金流向] Tushare {date} 失败: {e}")
    else:
        logger.info("[资金流向] Tushare 未配置, 用 akshare 回退")

    import akshare as ak
    if codes is None:
        from core.data.sources.tushare import get_all_stock_codes
        codes = get_all_stock_codes()
    total = len(codes)
    saved = 0
    lock = threading.Lock()
    done = [0]
    error_samples = []

    def _fetch_one(code):
        nonlocal saved
        try:
            mkt = "sh" if code.startswith("6") else "sz" if code.startswith(("0", "3")) else ""
            if not mkt:
                return
            import time as _time
            import random as _random
            _time.sleep(0.25 + _random.random() * 0.15)
            df = ak.stock_individual_fund_flow(stock=code, market=mkt)
            if df is None or df.empty:
                return
            date_col = next((c for c in df.columns if "日期" in str(c) or "时间" in str(c) or "date" in str(c).lower()), None)
            if not date_col:
                return
            df[date_col] = pd.to_datetime(df[date_col])
            latest = df[df[date_col] == df[date_col].max()]
            if latest.empty:
                return
            row = latest.iloc[-1]
            main_net = 0.0
            main_pct = 0.0
            turnover = 0.0
            for c in df.columns:
                cn = str(c)
                if ("主力净流入" in cn or "主力净额" in cn or "净流入" in cn) and "占比" not in cn:
                    main_net = float(row.get(c, 0) or 0)
                elif "主力" in cn and ("占比" in cn or "%" in cn):
                    main_pct = float(row.get(c, 0) or 0)
                elif cn in ("成交额", "成交金额"):
                    turnover = float(row.get(c, 0) or 0)
            if main_net == 0:
                for c in df.columns:
                    cn = str(c)
                    if "超大单净流入" in cn and "占比" not in cn:
                        main_net += float(row.get(c, 0) or 0)
                    elif "大单净流入" in cn and "占比" not in cn:
                        main_net += float(row.get(c, 0) or 0)
            main_pct = round(max(-100.0, min(100.0, main_pct)), 4)
            Database.execute(
                "INSERT INTO capital_flow_daily (code, trade_date, main_net_flow, main_net_pct, turnover) "
                "VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                "main_net_flow=VALUES(main_net_flow), main_net_pct=VALUES(main_net_pct), "
                "turnover=VALUES(turnover)",
                (code, date, main_net, main_pct, turnover),
            )
            with lock:
                nonlocal saved
                saved += 1
        except Exception as e:
            with lock:
                if len(error_samples) < 5:
                    error_samples.append(f"{code}: {e}")
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                err_info = ""
                if error_samples:
                    err_info = f" | 错误示例: {'; '.join(error_samples[:3])}"
                logger.info(f"[资金流向] akshare回退 {done[0]}/{total} ({saved} 有效){err_info}")

    logger.info(f"[资金流向] akshare回退: {total} 只, date={date}")
    with thread_pool.default.borrow() as ex:
        list(ex.map(_fetch_one, codes, max_workers=2))
    logger.info(f"[资金流向] akshare 完成: {saved}/{total}")
    return saved


def backfill_capital_flow(start_date, end_date, codes=None, max_workers=8):
    """历史回补资金流向数据。跳过已有数据的日期。"""
    from core.data import Database
    if not Database.is_available():
        return []

    existing = set()
    try:
        rows = Database.fetchall(
            "SELECT DISTINCT trade_date FROM capital_flow_daily "
            "WHERE trade_date >= %s AND trade_date <= %s",
            (start_date, end_date),
        )
        existing = set(r["trade_date"] for r in rows if r["trade_date"])
    except Exception:
        pass

    todo = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5 and d not in existing:
            todo.append(d)
        d += datetime.timedelta(days=1)

    if not todo:
        logger.info("[资金流向] 无需回补, 数据已完整")
        return []

    results = []
    lock = threading.Lock()
    done = [0]
    total = len(todo)

    def _backfill_one(date):
        try:
            n = fetch_and_save_capital_flow(codes=codes, date=date, max_workers=1)
            return {"date": date.isoformat(), "count": n, "status": "ok"}
        except Exception as e:
            return {"date": date.isoformat(), "count": 0, "status": str(e)}
        finally:
            with lock:
                done[0] += 1
                if done[0] % 10 == 0:
                    logger.info(f"[资金流向] 回补进度 {done[0]}/{total}")

    logger.info(f"[资金流向] 回补 {total} 个日期 (3 并发)")
    with thread_pool.default.borrow() as ex:
        futures = {ex.submit(_backfill_one, d): d for d in todo}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["date"])
    ok = sum(1 for r in results if r["status"] == "ok" and r["count"] > 0)
    logger.info(f"[资金流向] 回补完成: {ok}/{total} 日有数据")
    return results


# ═══ 北向资金 ═══

def fetch_and_save_north_flow(date=None):
    """盘后采集北向资金个股数据到 north_flow 表

    date: 目标采集日期，为 None 则默认今天。
          仅支持当日采集（akshare 无历史接口），非当日自动跳过。
    """
    try:
        import akshare as ak
        today = today_date()
        if date is not None:
            if hasattr(date, 'date'):
                date = date.date()
            if isinstance(date, datetime.datetime):
                date = date.date()
            if date != today:
                logger.warning(f"[北向资金] API 不支持历史日期，跳过 (目标={date}, 今天={today})")
                return 0
        date = today

        if date > datetime.date(2024, 8, 16):
            logger.warning("[北向资金] 交易所自 2024-08-19 起停止日频个股持股披露，跳过并禁止将旧快照写入新日期")
            return 0

        from core.data import Database
        Database.execute("""
            CREATE TABLE IF NOT EXISTS `north_flow` (
              id INT AUTO_INCREMENT PRIMARY KEY,
              code VARCHAR(10) NOT NULL,
              trade_date DATE NOT NULL,
              net_flow DECIMAL(16,2),
              buy_amount DECIMAL(16,2) DEFAULT 0,
              sell_amount DECIMAL(16,2) DEFAULT 0,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              UNIQUE KEY uk_code_date (code, trade_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        target_codes = _get_north_target_codes()
        logger.info(f"[北向资金] 开始采集 {len(target_codes)} 只个股...")
        saved = 0
        for code in target_codes:
            try:
                df = ak.stock_hsgt_individual_em(symbol=code)
                if df is None or df.empty:
                    continue
                latest = df.iloc[-1]
                net_col = next((c for c in df.columns if "增持资金" in str(c)), None)
                net = float(latest[net_col]) if net_col and pd.notna(latest.get(net_col)) else 0.0
                Database.execute(
                    "INSERT INTO north_flow (code, trade_date, net_flow) VALUES (%s,%s,%s) "
                    "ON DUPLICATE KEY UPDATE net_flow=VALUES(net_flow)",
                    (code, date, net))
                saved += 1
            except Exception:
                continue
        logger.info(f"[北向资金] {date} 采集完成: {saved}/{len(target_codes)} 只")
        return saved
    except Exception as e:
        logger.warning(f"[北向资金] {date} 采集失败: {e}")
        return 0


def _get_north_target_codes():
    """获取北向资金采集目标股票列表 (沪深300成分股)"""
    try:
        from core.data.sources.tushare import get_index_components
        codes = get_index_components("000300")
        if codes and len(codes) >= 100:
            return codes
    except Exception:
        pass
    try:
        import akshare as ak
        df = ak.index_stock_cons_csindex("000300")
        if df is not None and not df.empty:
            col = next((c for c in df.columns if "成分券代码" in str(c) or "code" in str(c).lower()), df.columns[0])
            return [str(c).strip()[:6] for c in df[col] if len(str(c).strip()) >= 6]
    except Exception:
        pass
    return []


def backfill_north_flow(start_date, end_date):
    """北向资金 API 不支持历史日期回补"""
    logger.warning("[北向资金] API 不支持历史日期，无法回补。数据将从今日开始自动积累。")
    return [{"date": start_date.isoformat(), "count": 0, "status": "API不支持历史日期"}]


def fetch_and_save_fundamentals(codes=None, max_workers=4):
    from core.data import Database
    if not Database.is_available():
        return {"saved": 0, "tushare": 0, "akshare": 0}

    if codes is None:
        from core.data.sources.tushare import get_all_stock_codes
        codes = get_all_stock_codes()
    codes = list(codes or [])
    if not codes:
        return {"saved": 0, "tushare": 0, "akshare": 0}

    from core.data.sources.tushare import _rate_limit, get_pro
    pro = get_pro()
    if not pro:
        return {"saved": 0, "tushare": 0, "akshare": 0}

    fields = "ts_code,ann_date,end_date,roe,netprofit_yoy,or_yoy,debt_to_assets,eps,grossprofit_margin,roa,ebit,netprofit_margin,ocf_to_debt,roe_yoy,assets_turn,quick_ratio,profit_dedt,diluted2_eps,bps,cfps"
    sql = """
        INSERT INTO fundamental_data
            (code,report_date,ann_date,roe,net_profit_yoy,revenue_yoy,debt_to_assets,eps,
             grossprofit_margin,roa,ebit,netprofit_margin,ocf_to_debt,roe_yoy,assets_turn,
             quick_ratio,profit_dedt,diluted2_eps,bps,cfps)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            ann_date=COALESCE(VALUES(ann_date),ann_date),
            roe=COALESCE(VALUES(roe),roe),net_profit_yoy=COALESCE(VALUES(net_profit_yoy),net_profit_yoy),
            revenue_yoy=COALESCE(VALUES(revenue_yoy),revenue_yoy),debt_to_assets=COALESCE(VALUES(debt_to_assets),debt_to_assets),
            eps=COALESCE(VALUES(eps),eps),grossprofit_margin=COALESCE(VALUES(grossprofit_margin),grossprofit_margin),
            roa=COALESCE(VALUES(roa),roa),ebit=COALESCE(VALUES(ebit),ebit),
            netprofit_margin=COALESCE(VALUES(netprofit_margin),netprofit_margin),
            ocf_to_debt=COALESCE(VALUES(ocf_to_debt),ocf_to_debt),roe_yoy=COALESCE(VALUES(roe_yoy),roe_yoy),
            assets_turn=COALESCE(VALUES(assets_turn),assets_turn),quick_ratio=COALESCE(VALUES(quick_ratio),quick_ratio),
            profit_dedt=COALESCE(VALUES(profit_dedt),profit_dedt),diluted2_eps=COALESCE(VALUES(diluted2_eps),diluted2_eps),
            bps=COALESCE(VALUES(bps),bps),cfps=COALESCE(VALUES(cfps),cfps)
    """

    def _safe_float(value):
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if _math.isfinite(result) else None

    def _date(value):
        if value is None:
            return None
        value = str(value).replace("-", "")[:8]
        try:
            return datetime.datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            return None

    batch_size = 1
    saved = 0
    failed_batches = 0
    end_date = today_date().strftime("%Y%m%d")
    for start in range(0, len(codes), batch_size):
        batch = codes[start:start + batch_size]
        ts_codes = ",".join(code if "." in code else (code + ".SH" if code.startswith("6") else code + ".SZ") for code in batch)
        try:
            _rate_limit()
            frame = pro.fina_indicator(
                ts_code=ts_codes,
                start_date="20180101",
                end_date=end_date,
                fields=fields,
                limit=100,
            )
            if frame is not None and len(frame) >= 100:
                raise RuntimeError("TuShare 单次返回达到100行上限，批次可能被截断")
            rows = []
            if frame is not None and not frame.empty:
                for row in frame.to_dict("records"):
                    report_date = _date(row.get("end_date"))
                    ann_date = _date(row.get("ann_date"))
                    code = str(row.get("ts_code") or "").split(".")[0]
                    if not code or report_date is None or ann_date is None:
                        continue
                    rows.append((
                        code, report_date, ann_date,
                        _safe_float(row.get("roe")), _safe_float(row.get("netprofit_yoy")),
                        _safe_float(row.get("or_yoy")), _safe_float(row.get("debt_to_assets")),
                        _safe_float(row.get("eps")), _safe_float(row.get("grossprofit_margin")),
                        _safe_float(row.get("roa")), _safe_float(row.get("ebit")),
                        _safe_float(row.get("netprofit_margin")), _safe_float(row.get("ocf_to_debt")),
                        _safe_float(row.get("roe_yoy")), _safe_float(row.get("assets_turn")),
                        _safe_float(row.get("quick_ratio")), _safe_float(row.get("profit_dedt")),
                        _safe_float(row.get("diluted2_eps")), _safe_float(row.get("bps")),
                        _safe_float(row.get("cfps")),
                    ))
            if rows:
                saved += Database.executemany(sql, rows) or 0
        except Exception as error:
            failed_batches += 1
            logger.warning("[基本面] TuShare 批次 %s 失败: %s", start // batch_size + 1, error)

    logger.info("[基本面] PIT 采集完成: %s 条；失败批次 %s；缺公告日行跳过", saved, failed_batches)
    return {"saved": saved, "tushare": saved, "akshare": 0, "failed_batches": failed_batches}


def _score_north_flow(code, df=None, north_flow_data=None, as_of=None):
    """北向资金评分：当日净流入/5日均成交额 + 近N日趋势 → -1..1
    as_of 给定时 DB 回退查询截断到该日, 防止历史回测用今日北向资金前视。"""
    try:
        net_flow = None
        recent_flows = []
        if north_flow_data is not None:
            val = north_flow_data.get(code)
            if isinstance(val, (list, tuple)):
                recent_flows = [(d, float(v)) for d, v in val if v is not None]
                if recent_flows:
                    net_flow = recent_flows[0][1]
            elif val is not None:
                net_flow = float(val)
        if net_flow is None:
            from core.data import Database
            if as_of is not None:
                rows = Database.fetchall(
                    "SELECT trade_date, net_flow FROM north_flow WHERE code=%s "
                    "AND trade_date <= %s ORDER BY trade_date DESC LIMIT 20",
                    (code, as_of),
                )
            else:
                rows = Database.fetchall(
                    "SELECT trade_date, net_flow FROM north_flow WHERE code=%s "
                    "AND trade_date <= CURDATE() ORDER BY trade_date DESC LIMIT 20",
                    (code,),
                )
            if rows:
                net_flow = float(rows[0]["net_flow"])
                recent_flows = [(r["trade_date"], float(r["net_flow"])) for r in rows]
        if net_flow is None or df is None or len(df) < 5:
            return 0.0
        target_date = as_of or today_date()
        if isinstance(target_date, str):
            target_date = datetime.date.fromisoformat(target_date[:10])
        elif isinstance(target_date, datetime.datetime):
            target_date = target_date.date()
        if recent_flows:
            observed_date = recent_flows[0][0]
            if isinstance(observed_date, str):
                observed_date = datetime.date.fromisoformat(observed_date[:10])
            elif isinstance(observed_date, datetime.datetime):
                observed_date = observed_date.date()
        else:
            observed_date = target_date
        if not isinstance(observed_date, datetime.date):
            return 0.0
        if (target_date - observed_date).days > 10 or observed_date > target_date:
            return 0.0
        avg_amount = float(df.tail(5)["amount"].mean())
        if avg_amount <= 0:
            return 0.0
        pct = net_flow / avg_amount * 100
        base = _math.tanh(pct / 12.0)
        consecutive = 0
        for _, f in recent_flows:
            if f > 0:
                consecutive += 1
            else:
                break
        if consecutive >= 5:
            base += 0.2
        elif consecutive >= 3:
            base += 0.1
        if len(recent_flows) >= 10:
            avg5 = sum(f for _, f in recent_flows[:5]) / 5
            avg20 = sum(f for _, f in recent_flows[:20]) / min(len(recent_flows), 20)
            if avg20 > 0 and avg5 > avg20:
                base += 0.1
            elif avg20 < 0 and avg5 < avg20:
                base -= 0.1
        return round(max(-1.0, min(1.0, base)), 3)
    except Exception:
        return 0.0


# ═══ 小市值/换手率因子 ═══
_size_cache = {}
_size_cache_lock = threading.Lock()
_size_cache_time = 0
_SIZE_CACHE_TTL = 300


def _size_date_key(eval_date):
    return str(eval_date) if eval_date else "now"


def _fetch_size_batch(codes, eval_date=None):
    """批量获取个股市值 — 按决策日 (eval_date) 取 point-in-time 市值, 防前视。
    eval_date 为空 (实时) 走 MySQL 最新 + akshare 回退;
    eval_date 给定时 (历史回测/滚动筛选) 走 daily_basic 当日/最近一条, 禁止用当前市值给过去打分。
    """
    global _size_cache_time
    now = time.time()
    if now - _size_cache_time > _SIZE_CACHE_TTL:
        with _size_cache_lock:
            _size_cache.clear()
            _size_cache_time = now

    date_key = _size_date_key(eval_date)
    missing = [c for c in codes if (date_key, c) not in _size_cache]
    if not missing:
        return

    if eval_date is not None:
        try:
            from core.domain.factors.market_cap import get_market_cap_map
            mv_map = get_market_cap_map(missing, eval_date)
            with _size_cache_lock:
                for c, mv in mv_map.items():
                    if mv > 0:
                        _size_cache[(date_key, c)] = mv * 10000  # 万元 → 元
        except Exception as e:
            logger.debug(f"[小市值] point-in-time 获取失败: {e}")
        return

    with _size_cache_lock:
        _size_cache.update({(date_key, c): v for c, v in _fetch_size_from_mysql(missing).items()})
    still_missing = [c for c in missing if (date_key, c) not in _size_cache]
    if not still_missing:
        return

    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        with _size_cache_lock:
            for row in df.itertuples(index=False):
                code = str(getattr(row, '代码', '') or '')
                if code in still_missing:
                    _size_cache[(date_key, code)] = float(getattr(row, '总市值', 0) or 0)
    except Exception as e:
        logger.debug(f"[小市值] akshare 获取失败: {e}")


def _fetch_size_from_mysql(codes):
    """从 MySQL daily_basic 取最新总市值（万元→元），按代码分组取最新日期"""
    result = {}
    try:
        from core.data import Database
        if not Database.is_available():
            return result
        batch_size = 500
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            ph = ','.join(['%s'] * len(batch))
            rows = Database.fetchall(
                f"SELECT t1.code, t1.total_mv FROM daily_basic t1 "
                f"INNER JOIN (SELECT code, MAX(trade_date) AS max_date FROM daily_basic "
                f"WHERE code IN ({ph}) GROUP BY code) t2 "
                f"ON t1.code = t2.code AND t1.trade_date = t2.max_date",
                tuple(batch))
            for r in (rows or []):
                code = str(r["code"]).strip()
                mv = float(r["total_mv"] or 0)
                if mv > 0 and code in codes:
                    result[code] = mv * 10000
    except Exception as e:
        logger.debug(f"[小市值] MySQL 获取失败: {e}")
    return result


def _score_size_turnover(code, df=None, eval_date=None):
    """小市值+换手率因子 → -1..1。
    小市值溢价 + 适度活跃（换手率2-10%最优），惩罚过低/过高换手。
    eval_date 用于取 point-in-time 市值, 防历史回测前视。
    """
    try:
        size = _size_cache.get((_size_date_key(eval_date), code), 0)
        if size <= 0:
            return 0.0

        if size >= 1e12:
            size_score = -1.0
        elif size >= 5e11:
            size_score = -0.5
        elif size >= 2e11:
            size_score = 0.0
        elif size >= 5e10:
            size_score = 0.5
        else:
            size_score = 1.0

        turnover = 0.0
        if df is not None and len(df) >= 1:
            t = df.iloc[-1].get("turnover", 0)
            if t and float(t) > 0:
                turnover = float(t)

        if turnover <= 0:
            turnover_score = 0.0
        elif 2 <= turnover <= 10:
            turnover_score = 1.0
        elif 1 <= turnover < 2 or 10 < turnover <= 15:
            turnover_score = 0.5
        elif turnover > 20:
            turnover_score = -0.5
        else:
            turnover_score = 0.0

        return round(max(-1.0, min(1.0, size_score * 0.6 + turnover_score * 0.4)), 3)
    except Exception:
        return 0.0


def _parse_num(text):
    """从财务文本提取数值（处理百分号、亿/万、负数等）"""
    if not text or text in ("nan", "None", ""):
        return None
    t = str(text).replace(",", "").replace(" ", "")
    is_pct = "%" in t
    t = t.replace("%", "")
    num = re.search(r"[-]?\d+\.?\d*", t)
    if not num:
        return None
    val = float(num.group())
    if "万" in t:
        val *= 10000
    elif "亿" in t:
        val *= 100000000
    return val / 100 if is_pct else val


# ═══ Amihud 非流动性因子 ═══

def _score_illiq(df, n=20, close_arr=None, ret_arr=None):
    """Amihud非流动性: abs(日收益)/成交额 的滚动均值 → -1..1。
    流动性越差(illiquidity越高) = 负分，流动性好 = 正分。"""
    try:
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        amounts = df.get("amount")
        if amounts is None:
            amounts = df["volume"].astype(float) * closes
        else:
            amounts = amounts.astype(float)
        if len(closes) < n + 1:
            return 0.0
        rets = (ret_arr if ret_arr is not None else closes.pct_change(fill_method=None)).dropna().iloc[-n:]
        amts = amounts.iloc[-n:]
        if len(rets) < n or amts.isna().all():
            return 0.0
        daily_illiq = rets.abs() / amts.replace(0, np.nan)
        daily_illiq = daily_illiq.dropna()
        if len(daily_illiq) < 5:
            return 0.0
        avg_illiq = daily_illiq.mean()
        # amount 单位元: 典型 Amihud ≈ |ret|/amount ∈ [2e-11(大盘), 2e-9(小盘)].
        # 1e8 使整个区间都趋近 0, 因子无区分度; 5e8 将小盘端映射到 ~-0.76, 大盘端 ~-0.01
        return round(_math.tanh(-avg_illiq * 5e8), 4)
    except Exception:
        return 0.0


# ═══ Overnight Gap 隔夜跳空因子 ═══

def _score_overnight_gap(df, n=20, close_arr=None):
    """隔夜跳空: (今开/昨收-1) 的均值+标准差 → -1..1。
    持续高开=强势(正分)，持续低开=弱势(负分)，波动过大=不稳定(扣分)。"""
    try:
        opens = df["open"].astype(float)
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        if len(opens) < n + 1 or len(closes) < n + 1:
            return 0.0
        import numpy as np
        gaps = (opens.iloc[-n:] / closes.shift(1).iloc[-n:] - 1).dropna()
        if len(gaps) < 5:
            return 0.0
        gap_mean = float(gaps.mean())
        gap_std = float(gaps.std())
        score = gap_mean * 5 - gap_std * 10
        return round(_math.tanh(score * 3), 4)
    except Exception:
        return 0.0


# ═══ 白盒因子: 行业残差动量 ═══

def _precompute_industry_returns(snapshot, industry_map, n=20):
    """预计算每个行业的 N 日平均涨幅，供残差动量因子复用。"""
    by_industry = {}
    for code, (df, name, close, volume) in snapshot.items():
        ind = industry_map.get(code, {}).get("industry", "")
        if not ind:
            continue
        closes = df["close"].astype(float)
        if len(closes) < n:
            continue
        ret = (closes.iloc[-1] / closes.iloc[-n] - 1)
        if ind not in by_industry:
            by_industry[ind] = []
        by_industry[ind].append(ret)
    return {ind: sum(rets) / max(len(rets), 1) for ind, rets in by_industry.items()}


def _score_industry_residual_momentum(code, df, n=20, industry_map=None, industry_returns=None, close_arr=None):
    """行业残差动量: 个股N日涨幅 - 同行业均值 → -1..1。"""
    try:
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        ret_i = (closes.iloc[-1] / closes.iloc[-n] - 1)
        ret_ind = 0.0
        if industry_returns:
            ind = industry_map.get(code, {}).get("industry", "") if industry_map else ""
            ret_ind = industry_returns.get(ind, 0.0)
        residual = ret_i - ret_ind
        return round(_math.tanh(residual * 3), 4)
    except Exception:
        return 0.0


# ═══ 白盒因子: 量价背离 ═══

def _score_volume_price_divergence(df, n=10, close_arr=None):
    """量价背离: 缩量上涨=看多, 放量下跌=看空 → -1..1。"""
    try:
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        volumes = df["volume"].astype(float)
        if len(closes) < n or len(volumes) < n:
            return 0.0
        price_chg = (closes.iloc[-1] / closes.iloc[-n] - 1)
        vol_recent = volumes.iloc[-n:].mean()
        vol_prior = volumes.iloc[-n*2:-n].mean() if len(volumes) >= n * 2 else volumes.iloc[:n].mean()
        vol_chg = (vol_recent / vol_prior - 1) if vol_prior > 0 else 0
        raw = price_chg * 5 - vol_chg * 3
        return round(_math.tanh(raw * 2), 4)
    except Exception:
        return 0.0


# ═══ 白盒因子: 资金异动 ═══

def _score_smart_money_flow(code, df, n=5, flow_data=None, close_arr=None):
    """资金异动: 主力净流入持续但股价滞后 → -1..1。"""
    try:
        if not flow_data:
            return 0.0
        flows = flow_data.get(code)
        if not flows or len(flows) < n:
            return 0.0
        recent_flows = flows[-n:]
        avg_flow = sum(recent_flows) / max(len(recent_flows), 1)
        closes = close_arr if close_arr is not None else df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        price_chg = (closes.iloc[-1] / closes.iloc[-n] - 1)
        base = _math.tanh(avg_flow * 4)
        bonus = 0.0
        if avg_flow > 0.5 and price_chg < 0.02:
            bonus = 0.3
        elif avg_flow > 1.0 and price_chg < 0.01:
            bonus = 0.5
        elif avg_flow < -0.5 and price_chg > 0.03:
            bonus = -0.3
        return round(max(-1.0, min(1.0, base + bonus)), 4)
    except Exception:
        return 0.0


def _score_etf_flow(code, industry=None, as_of=None, days=5):
    """ETF资金流评分: -1..1。
    根据股票所属行业ETF近N日资金净流入方向评分。
    无对应行业ETF → 0分（不惩罚）。
    """
    if not industry:
        return 0.0
    try:
        from core.data.sources.etf_flow import get_stock_etf_flow_score
        return get_stock_etf_flow_score(code, industry, as_of, days)
    except Exception:
        return 0.0


def _rank_sectors(eval_date=None, top_n=5):
    """板块轮动行业排序: 多维度综合评分 → Top-N行业。
    四维度: 行业动量(0.35) + ETF资金流(0.25) + 行业宽度(0.20) + 北向偏好(0.20)
    返回 [(行业名, 综合分, 细分维度dict), ...]
    """
    from core.data import Database

    ed = eval_date or _today_str()
    scores = {}

    # 1. 行业动量: 行业内个股近20日平均涨跌幅
    rows = Database.fetchall("""
        SELECT s.sector_name as industry, AVG(d.pct_chg) as avg_ret
        FROM daily_quotes d
        JOIN stock_sectors s ON d.code = s.code AND s.sector_type = 'industry'
        WHERE d.trade_date BETWEEN %s AND %s
        GROUP BY s.sector_name
        HAVING COUNT(DISTINCT d.code) >= 5
    """, (_lookback_start(ed, 20), ed))
    momentum = {}
    for r in rows:
        ind = r["industry"]
        if ind:
            momentum[ind] = float(r["avg_ret"] or 0)

    if momentum:
        max_abs = max(abs(v) for v in momentum.values()) or 1
        for ind in momentum:
            scores.setdefault(ind, {})["momentum"] = round(momentum[ind] / max_abs, 4)

    # 2. ETF资金流: 行业ETF近5日资金净流入
    try:
        from core.data.sources.etf_flow import get_etf_flow_by_industry
        etf_flows = get_etf_flow_by_industry(as_of=ed, days=5)
        if etf_flows:
            max_flow = max(abs(v) for v in etf_flows.values()) or 1
            for ind, v in etf_flows.items():
                scores.setdefault(ind, {})["etf_flow"] = round(v / max_flow, 4)
    except Exception:
        pass

    # 3. 行业宽度: 近10日正收益股票占比
    rows = Database.fetchall("""
        SELECT s.sector_name as industry,
               SUM(CASE WHEN (d.close - d10.close) / NULLIF(d10.close, 0) > 0 THEN 1 ELSE 0 END) / COUNT(*) as breadth
        FROM daily_quotes d
        JOIN stock_sectors s ON d.code = s.code AND s.sector_type = 'industry'
        LEFT JOIN daily_quotes d10 ON d.code = d10.code AND d10.trade_date = %s
        WHERE d.trade_date = %s
        GROUP BY s.sector_name
        HAVING COUNT(*) >= 5
    """, (_lookback_start(ed, 10), ed))
    for r in rows:
        ind = r["industry"]
        if ind:
            scores.setdefault(ind, {})["breadth"] = round(float(r["breadth"] or 0), 4)

    # 4. 北向偏好: 行业北向资金持股变化 (如有数据)
    try:
        rows = Database.fetchall("""
            SELECT s.sector_name as industry, AVG(n.hold_ratio_chg) as avg_chg
            FROM north_flow n
            JOIN stock_sectors s ON n.code = s.code AND s.sector_type = 'industry'
            WHERE n.trade_date BETWEEN %s AND %s
            GROUP BY s.sector_name
            HAVING COUNT(*) >= 3
        """, (_lookback_start(ed, 5), ed))
        north = {}
        for r in rows:
            ind = r["industry"]
            if ind and r["avg_chg"] is not None:
                north[ind] = float(r["avg_chg"])
        if north:
            max_abs = max(abs(v) for v in north.values()) or 1
            for ind, v in north.items():
                scores.setdefault(ind, {})["north_flow"] = round(v / max_abs, 4)
    except Exception:
        pass

    # 综合评分: 动量0.35 + ETF流0.25 + 宽度0.20 + 北向0.20
    weights = {"momentum": 0.35, "etf_flow": 0.25, "breadth": 0.20, "north_flow": 0.20}
    ranked = []
    for ind, dims in scores.items():
        composite = sum(weights.get(d, 0) * dims.get(d, 0) for d in dims)
        ranked.append((ind, round(composite, 4), dims))
    ranked.sort(key=lambda x: x[1], reverse=True)

    # 行业相关性去重: 同板块组只保留排名最高的
    deduped = _dedupe_sectors(ranked, top_n)
    return deduped


# ── 行业板块分组（用于去重：同组内只保留最高分行业）──
SECTOR_GROUPS = {
    "TMT": {"电子", "计算机", "通信", "传媒"},
    "金融": {"银行", "非银金融", "房地产"},
    "消费": {"食品饮料", "家用电器", "商贸零售", "社会服务", "美容护理", "纺织服饰", "轻工制造"},
    "周期": {"有色金属", "钢铁", "煤炭", "石油石化", "基础化工"},
    "制造": {"电力设备", "机械设备", "汽车", "国防军工"},
    "基建": {"建筑材料", "建筑装饰", "公用事业", "交通运输", "环保"},
    "医药": {"医药生物"},
    "农业": {"农林牧渔"},
}


def _find_group(industry):
    for g, members in SECTOR_GROUPS.items():
        if industry in members:
            return g
    return industry  # 未分组的行业各自独立


def _dedupe_sectors(ranked, top_n):
    """同板块组只保留得分最高的行业，空缺从后续补位"""
    seen_groups = set()
    result = []
    for ind, score, dims in ranked:
        g = _find_group(ind)
        if g not in seen_groups:
            seen_groups.add(g)
            result.append((ind, score, dims))
        if len(result) >= top_n:
            break
    return result


def _rank_concept_sectors(eval_date=None, top_n=5):
    """概念板块轮动排序: 三维度综合评分 → Top-N概念板块。
    三维度: 概念动量(0.45) + 概念宽度(0.30) + 北向偏好(0.25)
    概念板块没有对应ETF，去掉ETF资金流维度。
    返回 [(概念名, 综合分, 细分维度dict), ...]
    """
    from core.data import Database

    ed = eval_date or _today_str()
    scores = {}

    # 1. 概念动量: 板块内个股近20日平均涨跌幅
    rows = Database.fetchall("""
        SELECT s.sector_name as industry, AVG(d.pct_chg) as avg_ret
        FROM daily_quotes d
        JOIN stock_sectors s ON d.code = s.code AND s.sector_type = 'concept'
        WHERE d.trade_date BETWEEN %s AND %s
        GROUP BY s.sector_name
        HAVING COUNT(DISTINCT d.code) >= 3
    """, (_lookback_start(ed, 20), ed))
    momentum = {}
    for r in rows:
        ind = r["industry"]
        if ind:
            momentum[ind] = float(r["avg_ret"] or 0)

    if momentum:
        max_abs = max(abs(v) for v in momentum.values()) or 1
        for ind in momentum:
            scores.setdefault(ind, {})["momentum"] = round(momentum[ind] / max_abs, 4)

    # 2. 概念宽度: 近10日正收益股票占比
    rows = Database.fetchall("""
        SELECT s.sector_name as industry,
               SUM(CASE WHEN (d.close - d10.close) / NULLIF(d10.close, 0) > 0 THEN 1 ELSE 0 END) / COUNT(*) as breadth
        FROM daily_quotes d
        JOIN stock_sectors s ON d.code = s.code AND s.sector_type = 'concept'
        LEFT JOIN daily_quotes d10 ON d.code = d10.code AND d10.trade_date = %s
        WHERE d.trade_date = %s
        GROUP BY s.sector_name
        HAVING COUNT(*) >= 3
    """, (_lookback_start(ed, 10), ed))
    for r in rows:
        ind = r["industry"]
        if ind:
            scores.setdefault(ind, {})["breadth"] = round(float(r["breadth"] or 0), 4)

    # 3. 北向偏好: 板块北向资金持股变化 (如有数据)
    try:
        rows = Database.fetchall("""
            SELECT s.sector_name as industry, AVG(n.hold_ratio_chg) as avg_chg
            FROM north_flow n
            JOIN stock_sectors s ON n.code = s.code AND s.sector_type = 'concept'
            WHERE n.trade_date BETWEEN %s AND %s
            GROUP BY s.sector_name
            HAVING COUNT(*) >= 2
        """, (_lookback_start(ed, 5), ed))
        north = {}
        for r in rows:
            ind = r["industry"]
            if ind and r["avg_chg"] is not None:
                north[ind] = float(r["avg_chg"])
        if north:
            max_abs = max(abs(v) for v in north.values()) or 1
            for ind, v in north.items():
                scores.setdefault(ind, {})["north_flow"] = round(v / max_abs, 4)
    except Exception:
        pass

    weights = {"momentum": 0.45, "breadth": 0.30, "north_flow": 0.25}
    ranked = []
    for ind, dims in scores.items():
        composite = sum(weights.get(d, 0) * dims.get(d, 0) for d in dims)
        ranked.append((ind, round(composite, 4), dims))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def _today_str():
    from core.shared.persistence import today_date
    return today_date().isoformat()


def _lookback_start(end_str, window_days):
    """返回 window_days*2 天前的日期，确保 SQL 查询有足够数据覆盖窗口"""
    from datetime import timedelta, date
    from core.shared.persistence import today_date
    if end_str is None:
        return (today_date() - timedelta(days=max(1, window_days or 10) * 2)).isoformat()
    if window_days is None or window_days <= 0:
        window_days = 10
    d = date.fromisoformat(end_str) if isinstance(end_str, str) else end_str
    return (d - timedelta(days=window_days * 2)).isoformat()
