import pandas as pd
import numpy as np


class MarketBreadth:

    _db_class = None

    @classmethod
    def _db(cls):
        if cls._db_class is None:
            from core.data import Database
            cls._db_class = Database
        return cls._db_class

    def compute_daily(self, date):
        rows = self._db().fetchall(
            "SELECT code, pct_chg, volume, amount, turnover, high, low, close "
            "FROM daily_quotes WHERE trade_date = %s",
            (date,)
        )
        if not rows:
            return None

        try:
            prev_rows = self._db().fetchall(
                "SELECT current_quote.code, prior_quote.close FROM daily_quotes current_quote "
                "LEFT JOIN daily_quotes prior_quote ON prior_quote.code=current_quote.code "
                "AND prior_quote.trade_date=(SELECT MAX(p.trade_date) FROM daily_quotes p "
                "WHERE p.code=current_quote.code AND p.trade_date<current_quote.trade_date) "
                "WHERE current_quote.trade_date=%s",
                (date,),
            )
            prev_close = {r["code"]: r["close"] for r in prev_rows}
            for row in rows:
                try:
                    pct = float(row["pct_chg"]) if row.get("pct_chg") is not None else None
                    if pct is not None and np.isfinite(pct) and pct != 0:
                        continue
                    current = float(row["close"])
                    previous = float(prev_close.get(row["code"]))
                    if np.isfinite(current) and np.isfinite(previous) and current > 0 and previous > 0:
                        row["pct_chg"] = round((current - previous) / previous * 100, 2)
                    else:
                        row["pct_chg"] = None
                except (TypeError, ValueError, OverflowError):
                    row["pct_chg"] = None
        except Exception:
            for row in rows:
                if row.get("pct_chg") == 0:
                    row["pct_chg"] = None

        result = self.compute_from_rows(rows)

        # 补充 20日新高/新低 — 需要历史数据
        if result:
            try:
                nh_nl = self._compute_new_high_low_daily(date)
                if nh_nl:
                    result.update(nh_nl)
            except Exception:
                pass
        return result

    def _compute_new_high_low_daily(self, date, lookback=20):
        """查询过去 lookback 个交易日, 统计当日创 N 日新高/新低的股票数"""
        rows = self._db().fetchall(
            "SELECT code, high, low FROM daily_quotes WHERE trade_date = %s",
            (date,)
        )
        if not rows or len(rows) < 100:
            return None
        today_high = {r["code"]: r["high"] for r in rows if r["high"] is not None}
        today_low = {r["code"]: r["low"] for r in rows if r["low"] is not None}
        codes = list(today_high.keys() & today_low.keys())
        if len(codes) < 100:
            return None

        past_dates = self._db().fetchall(
            "SELECT DISTINCT trade_date FROM daily_quotes "
            "WHERE trade_date < %s ORDER BY trade_date DESC LIMIT %s",
            (date, lookback)
        )
        if not past_dates:
            return None
        past_start = past_dates[-1]["trade_date"]
        past_end = past_dates[0]["trade_date"]

        past_rows = self._db().fetchall(
            "SELECT code, MAX(high) as max_high, MIN(low) as min_low "
            "FROM daily_quotes WHERE trade_date BETWEEN %s AND %s "
            "GROUP BY code",
            (past_start, past_end)
        )
        past_high = {r["code"]: r["max_high"] for r in past_rows if r["max_high"] is not None}
        past_low = {r["code"]: r["min_low"] for r in past_rows if r["min_low"] is not None}

        new_high = sum(1 for c in codes if c in past_high and today_high[c] > past_high[c])
        new_low = sum(1 for c in codes if c in past_low and today_low[c] < past_low[c])
        return {"new_high_approx": new_high, "new_low_approx": new_low}

    _history_cache = {}
    _history_cache_key = None

    def compute_history(self, start, end):
        cache_key = (str(start)[:10], str(end)[:10])
        if cache_key == self._history_cache_key and self._history_cache:
            return self._history_cache.get("df", pd.DataFrame()).copy()

        sql = (
            "SELECT trade_date, code, pct_chg, volume, amount, turnover, high, low, close "
            "FROM daily_quotes WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, code"
        )
        rows = self._db().fetchall(sql, (start, end))
        if not rows:
            return pd.DataFrame()

        previous_close = {}
        for row in sorted(rows, key=lambda item: (item.get("code", ""), str(item.get("trade_date", "")))):
            code = row.get("code")
            try:
                current_close = float(row.get("close"))
                pct = float(row["pct_chg"]) if row.get("pct_chg") is not None else None
                if pct is None or not np.isfinite(pct) or pct == 0:
                    prior = previous_close.get(code)
                    row["pct_chg"] = round((current_close / prior - 1) * 100, 2) if prior and np.isfinite(current_close) and current_close > 0 else None
                if np.isfinite(current_close) and current_close > 0:
                    previous_close[code] = current_close
            except (TypeError, ValueError, OverflowError):
                row["pct_chg"] = None

        try:
            import polars as pl
        except Exception:
            df = pd.DataFrame(rows)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values(["code", "trade_date"])
            df["roll_high_20"] = df.groupby("code")["high"].transform(
                lambda s: s.shift(1).rolling(20, min_periods=5).max()
            )
            df["roll_low_20"] = df.groupby("code")["low"].transform(
                lambda s: s.shift(1).rolling(20, min_periods=5).min()
            )
            df["is_new_high"] = (df["high"] > df["roll_high_20"]).astype(int)
            df["is_new_low"] = (df["low"] < df["roll_low_20"]).astype(int)
            results = []
            for trade_date, group in df.groupby("trade_date", sort=True):
                result = self.compute_from_rows(group.to_dict("records"))
                if result:
                    result["trade_date"] = trade_date
                    result["new_high_approx"] = int(group["is_new_high"].sum())
                    result["new_low_approx"] = int(group["is_new_low"].sum())
                    results.append(result)
            if not results:
                return pd.DataFrame()
            return pd.DataFrame(results).set_index("trade_date")
        df = pl.from_dicts(rows)
        dates = sorted(df["trade_date"].unique().to_list())

        # 20日新高/新低 — Polars rolling 替代 pandas pivot_table
        nh_nl_map = {}
        if len(dates) >= 2:
            roll = (
                df.sort(["code", "trade_date"])
                .with_columns(
                    pl.col("high").cast(pl.Float64).shift(1).rolling_max(20, min_periods=5).over("code").alias("roll_high_20"),
                    pl.col("low").cast(pl.Float64).shift(1).rolling_min(20, min_periods=5).over("code").alias("roll_low_20"),
                )
                .with_columns(
                    (pl.col("high") > pl.col("roll_high_20")).cast(pl.Int32).alias("is_new_high"),
                    (pl.col("low") < pl.col("roll_low_20")).cast(pl.Int32).alias("is_new_low"),
                )
            )
            nh_nl_agg = roll.group_by("trade_date").agg([
                pl.col("is_new_high").sum().alias("nh"),
                pl.col("is_new_low").sum().alias("nl"),
            ])
            nh_nl_map = {r["trade_date"]: (r["nh"], r["nl"]) for r in nh_nl_agg.to_dicts()}

        results = []
        grouped = df.group_by("trade_date", maintain_order=True)
        for d, grp in grouped:
            day_rows = grp.to_dicts()
            r = self.compute_from_rows(day_rows)
            if r:
                r["trade_date"] = d[0] if isinstance(d, tuple) else d
                if d in nh_nl_map:
                    r["new_high_approx"], r["new_low_approx"] = nh_nl_map[d]
                results.append(r)
        if not results:
            return pd.DataFrame()
        result = pd.DataFrame(results).set_index("trade_date")
        self._history_cache_key = cache_key
        self._history_cache = {"df": result.copy()}
        return result

    def compute_from_rows(self, rows):
        sample_rows = [row for row in rows if isinstance(row, dict)]
        if not sample_rows:
            return None

        required_fields = ("volume", "amount", "high", "low", "close")
        valid_rows = [row for row in sample_rows if all(row.get(field) is not None for field in required_fields)]
        if len(valid_rows) < 100:
            return None

        pct = np.array([row.get("pct_chg") for row in valid_rows], dtype=float)
        valid_pct = np.isfinite(pct)
        has_pct = valid_pct.any()
        amount = np.array([row["amount"] for row in valid_rows], dtype=float)
        turnover_raw = np.array([row.get("turnover") for row in valid_rows], dtype=float)
        turnover = turnover_raw[~np.isnan(turnover_raw)]
        high = np.array([row["high"] for row in valid_rows], dtype=float)
        low = np.array([row["low"] for row in valid_rows], dtype=float)
        close = np.array([row["close"] for row in valid_rows], dtype=float)

        total = len(valid_rows)

        if has_pct:
            up_mask = (pct > 0) & valid_pct
            down_mask = (pct < 0) & valid_pct
            flat_mask = (pct == 0) & valid_pct
            up = int(up_mask.sum())
            down = int(down_mask.sum())
            flat = int(flat_mask.sum())
        else:
            up = down = flat = 0
        unknown = total - up - down - flat
        valid_change_count = up + down + flat

        up_ratio = up / (up + down) if (up + down) > 0 else None
        valid_amount = valid_pct & np.isfinite(amount) & (amount >= 0)
        known_amount = amount[valid_amount].sum()
        up_volume_ratio = float(amount[up_mask & valid_amount].sum() / known_amount) if known_amount > 0 else None
        valid_range = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (close > 0)
        amplitudes = (high[valid_range] - low[valid_range]) / close[valid_range] * 100
        observed_pct = pd.to_numeric(
            pd.Series([row.get("pct_chg") for row in sample_rows]), errors="coerce"
        ).to_numpy(dtype=float)
        observed_valid_pct = np.isfinite(observed_pct)
        observed_up = int(((observed_pct > 0) & observed_valid_pct).sum())
        observed_down = int(((observed_pct < 0) & observed_valid_pct).sum())
        observed_flat = int(((observed_pct == 0) & observed_valid_pct).sum())
        observed_known = observed_up + observed_down + observed_flat
        observed_amount = pd.to_numeric(
            pd.Series([row.get("amount") for row in sample_rows]), errors="coerce"
        ).to_numpy(dtype=float)
        observed_valid_amount = observed_valid_pct & np.isfinite(observed_amount) & (observed_amount >= 0)
        observed_known_amount = observed_amount[observed_valid_amount].sum()
        observed_up_volume_ratio = (
            float(observed_amount[(observed_pct > 0) & observed_valid_amount].sum() / observed_known_amount)
            if observed_known_amount > 0 else None
        )
        observed_high = pd.to_numeric(pd.Series([row.get("high") for row in sample_rows]), errors="coerce").to_numpy(dtype=float)
        observed_low = pd.to_numeric(pd.Series([row.get("low") for row in sample_rows]), errors="coerce").to_numpy(dtype=float)
        observed_close = pd.to_numeric(pd.Series([row.get("close") for row in sample_rows]), errors="coerce").to_numpy(dtype=float)
        observed_valid_range = (
            np.isfinite(observed_high) & np.isfinite(observed_low) & np.isfinite(observed_close)
            & (observed_close > 0)
        )
        observed_amplitudes = (
            (observed_high[observed_valid_range] - observed_low[observed_valid_range])
            / observed_close[observed_valid_range] * 100
        )
        limit_up = int(((pct >= 9.5) & valid_pct).sum()) if has_pct else None
        limit_down = int(((pct <= -9.5) & valid_pct).sum()) if has_pct else None

        # 20日新高/新低 — 需要历史数据, 由 compute_daily / compute_history 补充
        new_high = 0
        new_low = 0

        return {
            "up_count": up,
            "down_count": down,
            "flat_count": flat,
            "unknown_count": unknown,
            "valid_change_count": valid_change_count,
            "change_coverage": round(valid_change_count / total, 4) if total else None,
            "coverage_status": "unavailable" if not valid_change_count else "row_partial" if unknown else "row_complete",
            "coverage_basis": "daily_quotes_rows",
            "up_ratio": round(up_ratio, 4) if up_ratio is not None else None,
            "up_volume_ratio": round(up_volume_ratio, 4) if up_volume_ratio is not None else None,
            "new_high_approx": new_high,
            "new_low_approx": new_low,
            "pct_median": round(float(np.median(pct[valid_pct])), 2) if has_pct else None,
            "pct_std": round(float(np.std(pct[valid_pct])), 2) if has_pct else None,
            "amplitude_median": round(float(np.median(amplitudes)), 2) if len(amplitudes) else None,
            "turnover_median": round(float(np.median(turnover)), 2) if len(turnover) else None,
            "total_stocks": total,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "observed_row_count": len(sample_rows),
            "observed_up_count": observed_up,
            "observed_down_count": observed_down,
            "observed_flat_count": observed_flat,
            "observed_unknown_count": len(sample_rows) - observed_known,
            "observed_valid_change_count": observed_known,
            "observed_change_coverage": round(observed_known / len(sample_rows), 4) if sample_rows else None,
            "observed_up_ratio": round(observed_up / (observed_up + observed_down), 4)
            if observed_up + observed_down else None,
            "observed_up_volume_ratio": observed_up_volume_ratio,
            "observed_pct_median": round(float(np.median(observed_pct[observed_valid_pct])), 2)
            if observed_valid_pct.any() else None,
            "observed_amplitude_median": round(float(np.median(observed_amplitudes)), 2)
            if len(observed_amplitudes) else None,
        }


_breadth = MarketBreadth()


def get_breadth(date=None):
    """获取单日市场宽度。date 为 None 时取最新可用交易日。"""
    if date is None:
        from core.data import Database
        max_row = Database.fetchall(
            "SELECT MAX(trade_date) as d FROM daily_quotes"
        )
        if max_row and max_row[0]["d"]:
            date = str(max_row[0]["d"])
        else:
            return None  # DB 不可用，无法确定最新交易日，由调用方显式传入 date
    return _breadth.compute_daily(str(date))


def get_breadth_history(start, end):
    """获取历史市场宽度序列"""
    return _breadth.compute_history(start, end)


# ── P3: 波动率/换手率牛熊指标 ──

def compute_vol_turnover_kernel(df, lookback=200):
    """纯 L3 函数: 从已加载的指数日线 DataFrame 计算波动率/换手率牛熊指标。
    kernel = rolling_std(returns, lookback) / rolling_mean(turnover_rate, lookback)
    指标上升=熊市信号, 下降=牛市信号。
    返回 (latest_value, series_df) 或 (None, None)。
    """
    if df.empty or len(df) < lookback:
        return None, None
    close_col = "close" if "close" in df.columns else df.columns[2]
    df = df.copy()
    df["ret"] = df[close_col].pct_change(fill_method=None)
    if "turnover" not in df.columns or df["turnover"].isna().all():
        return None, None
    df["vol_std"] = df["ret"].rolling(lookback).std()
    df["turnover_ma"] = df["turnover"].rolling(lookback).mean()
    df["kernel"] = df["vol_std"] / df["turnover_ma"].replace(0, np.nan)
    latest = float(df["kernel"].dropna().iloc[-1]) if not df["kernel"].dropna().empty else None
    return latest, df[["kernel"]].dropna()


def get_vol_turnover_ratio(index_code="000300", lookback=200):
    """获取华泰牛熊指标 — 取数据后委托纯函数计算。"""
    from core.data.fetcher import DataFetcher
    fetcher = DataFetcher()
    df = fetcher.get_index_daily(index_code, days=lookback + 60)
    return compute_vol_turnover_kernel(df, lookback)


# ── P5: RSRS动量风格强度评分 ──

RSRS_STYLE_INDICES = {
    "上证50": "000016",
    "沪深300": "000300",
    "中证500": "000905",
    "国证2000": "399303",
}


def compute_rsrs_momentum(lows, lookback=89):
    """纯 L3 函数: 从已加载的最低价数组计算单个指数的 RSRS 动量评分。
    score = annualized_returns * r_squared
    返回 (score, annualized_ret, r2) 三元组，失败返回 (None, None, None)。
    """
    if lows is None or len(lows) < lookback:
        return None, None, None
    try:
        arr = np.array(lows[-lookback:], dtype=float)
        x = np.arange(lookback)
        y = np.log(arr)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        annualized_ret = np.exp(slope * 250) - 1
        score = float(annualized_ret * max(r2, 0))
        return round(score, 6), round(annualized_ret, 6), round(r2, 4)
    except Exception:
        return None, None, None


def get_rsrs_momentum_scores(lookback=89, as_of=None):
    """对4个规模指数计算RSRS动量评分—取数据后委托纯函数计算。
    as_of 给定时把指数数据截断到该日, 防止历史回测用未来 RSRS 调节权重。"""
    from core.data.fetcher import DataFetcher
    fetcher = DataFetcher()
    results = {}
    best_score = -float("inf")
    for name, code in RSRS_STYLE_INDICES.items():
        try:
            df = fetcher.get_index_daily(code, days=lookback + 30)
            if as_of is not None and "date" in df.columns:
                df = df[df["date"] <= pd.Timestamp(as_of)]
            if df.empty or len(df) < lookback:
                results[name] = {"score": None, "error": "数据不足"}
                continue
            low_col = "low" if "low" in df.columns else df.columns[3]
            lows = df[low_col].values.astype(float)
            score, ann_ret, r2 = compute_rsrs_momentum(lows, lookback)
            if score is None:
                results[name] = {"score": None, "error": "数据不足"}
            else:
                results[name] = {"score": score, "annualized_ret": ann_ret, "r2": r2}
                if score > best_score:
                    best_score = score
        except Exception as e:
            results[name] = {"score": None, "error": str(e)}
    best_rounded = round(best_score, 6)
    for name in results:
        if results[name].get("score") is not None:
            results[name]["dominant"] = (abs(results[name]["score"] - best_rounded) < 1e-9)
    return results
