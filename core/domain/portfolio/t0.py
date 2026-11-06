"""
T0Engine — T+0 做T建议引擎 (L3 纯计算)

从 JoinQuant buy_first_t 策略精简而来，输出两个档位：
  I 组 = 激进：满额做T，无 MA 过滤
  N 组 = 保守：MA20/MA60 破位时 T 量按 swing_factor 缩放（0.75/0.25），并给出风险提示

职责: 接收 5分钟 OHLC DataFrame + 日K列表 + 当前价/持仓/现金 → 计算低吸带/止损/止盈/建议T量。
禁调: DB、HTTP、datetime.now()/date.today()/time.time()。
"""
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from core.shared.minute_data import normalize_trade_quantity

# 与 JoinQuant 策略一致的参数（可外覆）
DEFAULT_CONFIG = {
    "max_t_ratio": 0.30,
    "entry_atr_multiple": 1.20,
    "entry_daily_atr_multiple": 0.35,
    "stop_atr_buffer": 0.25,
    "reward_risk_multiple": 1.50,
    "target_atr_multiple": 1.00,
    "max_stop_loss": 0.008,
    "emergency_stop_loss": 0.010,
    "min_net_profit_rate": 0.0020,
    "slippage_rate": 0.0001,
    "min_vol_percentile": 0.20,
    "max_vol_percentile": 0.85,
    "min_daily_range": 0.012,
    "max_low_position": 0.25,
    "max_confirm_rsi": 45,
    "swing_ma_fast": 20,
    "swing_ma_slow": 60,
    "swing_hold_fast": 0.75,
    "swing_hold_slow": 0.25,
    "min_daily_ma60_gap": 0.0,
    "min_ma20_slope": 0.0,
    "target_trade_value": 15000.0,
}


def _rsi(closes: pd.Series, period: int) -> float:
    delta = closes.diff()
    gain = float(delta.clip(lower=0).tail(period).mean())
    loss = float(-delta.clip(upper=0).tail(period).mean())
    if loss <= 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def _percentile_rank_last(values: pd.Series) -> float:
    latest = float(values.iloc[-1])
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0 or not np.isfinite(latest):
        return 0.0
    return float((clean <= latest).sum()) / float(len(clean))


class T0Engine:
    def __init__(self, config: Optional[Dict] = None):
        self.cfg = dict(DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)

    def compute(
        self,
        code: str,
        df_5min: pd.DataFrame,
        daily_bars: List[Dict],
        current_price: Optional[float] = None,
        total_shares: Optional[int] = None,
        available_cash: Optional[float] = None,
        sellable_shares: Optional[int] = None,
        quantity_status="unknown",
        semantic_status="unverified",
    ) -> Dict:
        """返回 {code, current_price, vwap, day_high, day_low, atr_abs, rsi,
        swing_factor, daily_ok, intraday_ok, aggressive:{...}, conservative:{...}}"""
        cfg = self.cfg
        out = {
            "code": code,
            "current_price": None,
            "vwap": None,
            "day_high": None,
            "day_low": None,
            "atr_abs": None,
            "rsi": None,
            "swing_factor": 1.0,
            "daily_ok": False,
            "intraday_ok": False,
            "quantity_status": "unknown",
            "semantic_status": "unverified",
            "daily_flags": [],
            "aggressive": None,
            "conservative": None,
        }

        if quantity_status != "verified" or semantic_status != "verified":
            out["daily_flags"].append("分钟量额或来源语义未核验")
            return out

        daily = self._build_daily(daily_bars)
        if daily is None or len(daily) < 60:
            out["daily_flags"].append("日K数据不足")
            return out
        daily_values = daily[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        daily_prices = daily[["open", "high", "low", "close"]]
        if (
            not np.isfinite(daily_values).all()
            or (daily_prices <= 0).any().any()
            or (daily["volume"] < 0).any()
            or (daily["high"] < daily[["open", "close"]].max(axis=1)).any()
            or (daily["low"] > daily[["open", "close"]].min(axis=1)).any()
            or (daily["high"] < daily["low"]).any()
        ):
            out["daily_flags"].append("日K价格或数量数据无效")
            return out

        daily_ok, daily_flags, metrics = self._daily_filter(daily)
        out["daily_ok"] = daily_ok
        out["daily_flags"] = daily_flags
        out.update(metrics)

        if df_5min is None or getattr(df_5min, "empty", True) or len(df_5min) < 6:
            out["daily_flags"].append("5分钟数据不足")
            return out
        cols = ["open", "high", "low", "close", "volume", "amount"]
        if any(col not in df_5min.columns for col in cols):
            out["daily_flags"].append("分钟量额字段缺失")
            return out
        bars = df_5min[cols].apply(pd.to_numeric, errors="coerce")
        price_values = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if (bars.empty or not np.isfinite(price_values).all() or
                (bars[["open", "high", "low", "close"]] <= 0).any().any()):
            out["daily_flags"].append("分钟价格数据无效")
            return out
        quantities = [
            normalize_trade_quantity(volume, amount)
            for volume, amount in zip(bars["volume"], bars["amount"])
        ]
        if any(item is None for item in quantities):
            out["daily_flags"].append("分钟量额数值无效")
            return out
        bars["volume"] = [item[0] for item in quantities]
        bars["amount"] = [item[1] for item in quantities]
        out["quantity_status"] = "verified"
        out["semantic_status"] = semantic_status

        minute_prices = bars[["open", "high", "low", "close"]]
        if (
            (minute_prices <= 0).any().any()
            or (bars["high"] < bars[["open", "close"]].max(axis=1)).any()
            or (bars["low"] > bars[["open", "close"]].min(axis=1)).any()
            or (bars["high"] < bars["low"]).any()
        ):
            out["daily_flags"].append("分钟OHLC价格关系无效")
            return out

        if current_price is None:
            price = float(bars["close"].iloc[-1])
        else:
            try:
                if isinstance(current_price, bool):
                    raise ValueError
                price = float(current_price)
            except (TypeError, ValueError, OverflowError):
                out["daily_flags"].append("当前价格无效")
                return out
            if not np.isfinite(price) or price <= 0:
                out["daily_flags"].append("当前价格无效")
                return out
        day_high = float(bars["high"].max())
        day_low = float(bars["low"].min())
        if "amount" in bars and float(bars["amount"].sum()) > 0:
            vwap = float(bars["amount"].sum() / max(bars["volume"].sum(), 1))
        else:
            vwap = float((bars["close"] * bars["volume"]).sum() / max(bars["volume"].sum(), 1))

        atr_abs = self._atr_5m(bars)
        rsi = _rsi(bars["close"], 7)

        cost_rate = cfg["min_net_profit_rate"] + 0.0013
        entry_distance = max(
            atr_abs * cfg["entry_atr_multiple"],
            price * metrics.get("daily_atr_pct", 0.0) * cfg["entry_daily_atr_multiple"],
            price * cost_rate,
        )
        entry_threshold = vwap - entry_distance
        low_position = (price - day_low) / max(day_high - day_low, price * 0.001)
        intraday_ok = (
            price <= entry_threshold
            and low_position <= cfg["max_low_position"]
            and rsi <= cfg["max_confirm_rsi"]
        )

        # 止损/止盈（与策略同口径）
        entry_price = price if intraday_ok else entry_threshold
        technical_stop = day_low - cfg["stop_atr_buffer"] * atr_abs
        fixed_stop = entry_price * (1 - cfg["max_stop_loss"])
        stop_price = max(fixed_stop, technical_stop)
        emergency_stop = entry_price * (1 - cfg["emergency_stop_loss"])
        risk_distance = max(entry_price - stop_price, 0.0)
        target_price = entry_price + max(
            risk_distance * cfg["reward_risk_multiple"],
            atr_abs * cfg["target_atr_multiple"],
            entry_price * cost_rate,
        )

        swing_factor = metrics.get("swing_factor", 1.0)
        try:
            if isinstance(available_cash, bool):
                raise ValueError
            cash_value = float(available_cash)
        except (TypeError, ValueError, OverflowError):
            cash_value = float("nan")
        cash_valid = np.isfinite(cash_value) and cash_value >= 0
        valid_sellable = (
            not isinstance(total_shares, bool)
            and isinstance(total_shares, int)
            and total_shares >= 0
            and not isinstance(sellable_shares, bool)
            and isinstance(sellable_shares, int)
            and 0 <= sellable_shares <= total_shares
        )
        if not valid_sellable:
            daily_flags.append("当日可卖股数缺失或无效，不能生成做T数量")
            shares = 0
            sell_shares = 0
        elif not cash_valid:
            daily_flags.append("本地可用资金缺失或无效，不能生成做T数量")
            shares = 0
            sell_shares = 0
        elif sellable_shares == 0:
            daily_flags.append("当日无可卖股数，无法完成T+0")
            shares = 0
            sell_shares = 0
        else:
            sell_shares = self._suggested_sell_shares(price, total_shares, sellable_shares)
            shares = min(
                self._suggested_shares(price, total_shares, cash_value),
                sell_shares,
            )
            sell_shares = min(sell_shares, shares)

        if semantic_status != "verified":
            daily_flags.append("分钟量额单位、累计口径或价格基准未核验，不提供执行建议")
            shares = 0
            sell_shares = 0

        semantic_verified = semantic_status == "verified"
        out.update({
            "current_price": round(price, 3) if semantic_verified else None,
            "vwap": round(vwap, 3) if semantic_verified else None,
            "day_high": round(day_high, 3) if semantic_verified else None,
            "day_low": round(day_low, 3) if semantic_verified else None,
            "atr_abs": round(atr_abs, 3) if semantic_verified else None,
            "rsi": round(rsi, 1) if semantic_verified else None,
            "swing_factor": swing_factor,
            "intraday_ok": intraday_ok if semantic_verified else False,
            "entry_threshold": round(entry_threshold, 3) if semantic_verified else None,
            "low_position": round(low_position, 3) if semantic_verified else None,
        })

        base = {
            "allowed": bool(daily_ok and intraday_ok and semantic_verified),
            "entry_price": round(entry_price, 3) if semantic_verified else None,
            "stop_price": round(stop_price, 3) if semantic_verified else None,
            "emergency_stop_price": round(emergency_stop, 3) if semantic_verified else None,
            "target_price": round(target_price, 3) if semantic_verified else None,
            "target_edge_pct": round((target_price / entry_price - 1) * 100, 3) if semantic_verified else None,
            "flags": list(daily_flags),
        }
        out["sellable_shares"] = sellable_shares
        out["aggressive"] = self._variant("I", "激进", 1.0, base, shares, sell_shares)
        out["conservative"] = self._variant("N", "保守", swing_factor, base, shares, sell_shares)
        return out

    # ── 内部 ────────────────────────────────────────────────
    def _build_daily(self, daily_bars):
        try:
            df = pd.DataFrame(daily_bars)
            if df.empty:
                return None
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_values("trade_date").reset_index(drop=True)
        except Exception:
            return None

    def _daily_filter(self, daily):
        cfg = self.cfg
        closes = daily["close"].astype(float)
        highs = daily["high"].astype(float)
        lows = daily["low"].astype(float)
        previous = closes.shift(1)
        true_range = pd.concat([
            highs - lows,
            (highs - previous).abs(),
            (lows - previous).abs(),
        ], axis=1).max(axis=1)
        ranges = ((highs - lows) / previous).dropna()
        median_range = float(ranges.tail(20).median()) if len(ranges) >= 20 else 0.0

        daily_atr_series = true_range.rolling(14).mean() / closes
        daily_atr_values = daily_atr_series.dropna().tail(60)
        daily_atr_pct = float(daily_atr_values.iloc[-1]) if len(daily_atr_values) else 0.0
        vol_percentile = _percentile_rank_last(daily_atr_values) if len(daily_atr_values) else 0.0

        ma20 = float(closes.tail(20).mean())
        prior_ma20 = float(closes.iloc[-25:-5].mean())
        ma60 = float(closes.tail(60).mean())
        ma20_gap = float(closes.iloc[-1] / ma20 - 1)
        ma60_gap = float(closes.iloc[-1] / ma60 - 1)
        ma20_slope = float(ma20 / prior_ma20 - 1)
        five_day_return = float(closes.iloc[-1] / closes.iloc[-6] - 1)

        swing_factor = 1.0
        if closes.iloc[-1] < ma60:
            swing_factor = cfg["swing_hold_slow"]
        elif closes.iloc[-1] < ma20:
            swing_factor = cfg["swing_hold_fast"]

        flags = []
        if median_range < cfg["min_daily_range"]:
            flags.append("日振幅不足")
        if vol_percentile < cfg["min_vol_percentile"]:
            flags.append("波动率偏低")
        if vol_percentile > cfg["max_vol_percentile"]:
            flags.append("波动率偏高")
        if ma60_gap < cfg["min_daily_ma60_gap"]:
            flags.append("跌破MA60")
        if ma20_slope < cfg["min_ma20_slope"]:
            flags.append("MA20走平/向下")
        if swing_factor < 1.0:
            flags.append(f"MA破位·T量降至{swing_factor:.0%}")

        daily_ok = not flags or (
            all(f not in flags for f in ("日振幅不足", "波动率偏低", "波动率偏高", "跌破MA60", "MA20走平/向下"))
        )

        metrics = {
            "median_range": round(median_range, 4),
            "daily_atr_pct": round(daily_atr_pct, 4),
            "vol_percentile": round(vol_percentile, 3),
            "ma20": round(ma20, 3),
            "ma60": round(ma60, 3),
            "ma20_slope": round(ma20_slope, 4),
            "ma60_gap": round(ma60_gap, 4),
            "five_day_return": round(five_day_return, 4),
            "swing_factor": swing_factor,
        }
        return daily_ok, flags, metrics

    def _atr_5m(self, bars):
        highs = bars["high"].astype(float)
        lows = bars["low"].astype(float)
        closes = bars["close"].astype(float)
        previous = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - previous).abs(),
            (lows - previous).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
        return atr if np.isfinite(atr) and atr > 0 else float(closes.iloc[-1] * 0.003)

    def _suggested_shares(self, price, total_shares, available_cash):
        cfg = self.cfg
        if not np.isfinite(price) or price <= 0:
            return 0
        if isinstance(available_cash, bool):
            return 0
        try:
            cash_value = float(available_cash)
        except (TypeError, ValueError, OverflowError):
            return 0
        if not np.isfinite(cash_value) or cash_value < 0:
            return 0
        cap = int(cfg["target_trade_value"] / price)
        if total_shares is not None:
            cap = min(cap, int(total_shares * cfg["max_t_ratio"]))
        cap = min(cap, int(cash_value / (price * 1.002)))
        return max(0, int(cap // 100 * 100))

    def _suggested_sell_shares(self, price, total_shares, sellable_shares):
        try:
            price_value = float(price)
        except (TypeError, ValueError, OverflowError):
            return 0
        if (not np.isfinite(price_value) or price_value <= 0 or
                isinstance(total_shares, bool) or not isinstance(total_shares, int) or
                isinstance(sellable_shares, bool) or not isinstance(sellable_shares, int) or
                total_shares < 0 or sellable_shares < 0 or sellable_shares > total_shares):
            return 0
        cfg = self.cfg
        cap = int(cfg["target_trade_value"] / price_value)
        cap = min(cap, int(total_shares * cfg["max_t_ratio"]), sellable_shares)
        return max(0, int(cap // 100 * 100))

    def _variant(self, name, label, factor, base, shares, sell_shares=0):
        v = dict(base)
        v["name"] = name
        v["label"] = label
        v["swing_factor"] = factor
        v["suggested_shares"] = int(shares * factor // 100 * 100) if shares else 0
        v["suggested_sell_shares"] = int(sell_shares * factor // 100 * 100) if sell_shares else 0
        v["allowed"] = bool(base["allowed"] and v["suggested_shares"] > 0)
        if factor < 1.0 and name == "N":
            v["flags"] = v["flags"] + [f"保守档：MA因子{factor:.0%}，T量减至{int(factor*100)}%"]
        return v
