"""
技术指标计算 + K 线形态识别
"""
from typing import Optional
import pandas as pd
import numpy as np
import ta


class Analyzer:
    """技术指标与分析工具"""

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """批量添加常用技术指标到 DataFrame"""
        if df.empty or len(df) < 20:
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 均线 ──
        df["MA5"] = ta.trend.sma_indicator(close, window=5)
        df["MA10"] = ta.trend.sma_indicator(close, window=10)
        df["MA20"] = ta.trend.sma_indicator(close, window=20)
        df["MA60"] = ta.trend.sma_indicator(close, window=60)

        df["EMA5"] = ta.trend.ema_indicator(close, window=5)
        df["EMA20"] = ta.trend.ema_indicator(close, window=20)

        # ── MACD ──
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        # ── RSI ──
        df["RSI14"] = ta.momentum.rsi(close, window=14)

        # ── KDJ ──
        low_min = low.rolling(window=9).min()
        high_max = high.rolling(window=9).max()
        rsv = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        df["KDJ_K"] = rsv.ewm(com=2).mean()
        df["KDJ_D"] = df["KDJ_K"].ewm(com=2).mean()
        df["KDJ_J"] = 3 * df["KDJ_K"] - 2 * df["KDJ_D"]

        # ── 布林带 ──
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

        # ── 成交量均线 ──
        df["VOL_MA5"] = volume.rolling(5).mean()
        df["VOL_MA20"] = volume.rolling(20).mean()

        # ── ATR (平均真实波幅) ──
        df["ATR14"] = ta.volatility.average_true_range(high, low, close, window=14)

        return df

    @staticmethod
    def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """regime 分类专用: 只算 MA/布林带/ATR, 避免全家桶冗余计算。

        与 add_indicators 中同名指标的计算方式完全一致 (原样复制 ta 调用),
        保证 regime 评分语义不变。
        """
        if df.empty or len(df) < 20:
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── 均线 (趋势评分用) ──
        df["MA5"] = ta.trend.sma_indicator(close, window=5)
        df["MA10"] = ta.trend.sma_indicator(close, window=10)
        df["MA20"] = ta.trend.sma_indicator(close, window=20)
        df["MA60"] = ta.trend.sma_indicator(close, window=60)

        # ── 布林带 (震荡评分用, 只保留上下轨) ──
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()

        # ── ATR (波动评分用) ──
        df["ATR14"] = ta.volatility.average_true_range(high, low, close, window=14)

        return df

    @staticmethod
    def latest_indicators(df: pd.DataFrame) -> dict:
        """提取最新一期的指标摘要"""
        if df.empty:
            return {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        def _v(key, default=0):
            val = latest.get(key)
            return float(val) if val is not None else float(default)

        return {
            "最新价": round(_v("close"), 2),
            "涨跌幅": round(_v("pct_chg"), 2),
            "MA5": round(_v("MA5"), 2),
            "MA10": round(_v("MA10"), 2),
            "MA20": round(_v("MA20"), 2),
            "MA60": round(_v("MA60"), 2),
            "MACD": round(_v("MACD"), 4),
            "MACD_signal": round(_v("MACD_signal"), 4),
            "RSI14": round(_v("RSI14"), 2),
            "KDJ_K": round(_v("KDJ_K"), 2),
            "KDJ_D": round(_v("KDJ_D"), 2),
            "KDJ_J": round(_v("KDJ_J"), 2),
            "BB_upper": round(_v("BB_upper"), 2),
            "BB_lower": round(_v("BB_lower"), 2),
            "ATR14": round(_v("ATR14"), 2),
            "成交量(万)": round(_v("volume") / 10000, 2),
        }

    @staticmethod
    def technical_signal_summary(df: pd.DataFrame, current_price=None) -> dict:
        if df is None or df.empty:
            return {}

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        def number(row, key):
            value = row.get(key)
            try:
                value = float(value)
                return value if np.isfinite(value) else None
            except (TypeError, ValueError):
                return None

        price = number(latest, "close")
        try:
            live_price = float(current_price)
            if np.isfinite(live_price) and live_price > 0:
                price = live_price
        except (TypeError, ValueError):
            pass

        items = []
        ma5 = number(latest, "MA5")
        ma10 = number(latest, "MA10")
        ma20 = number(latest, "MA20")
        if price and ma5 and ma10 and ma20:
            if price > ma5 > ma10 > ma20:
                status, score, detail = "bullish", 1, "现价与短中期均线呈多头排列"
            elif price < ma5 < ma10 < ma20:
                status, score, detail = "bearish", -1, "现价与短中期均线呈空头排列"
            elif price >= ma20:
                status, score, detail = "bullish", 1, "现价位于 MA20 上方，但均线尚未完全多头排列"
            else:
                status, score, detail = "bearish", -1, "现价位于 MA20 下方，中期趋势偏弱"
            items.append({
                "key": "ma", "label": "均线趋势", "status": status, "score": score,
                "value": f"{price:.2f} / MA20 {ma20:.2f}", "detail": detail,
            })

        macd = number(latest, "MACD")
        macd_signal = number(latest, "MACD_signal")
        prev_macd = number(previous, "MACD")
        prev_signal = number(previous, "MACD_signal")
        if macd is not None and macd_signal is not None:
            if prev_macd is not None and prev_signal is not None and prev_macd < prev_signal and macd >= macd_signal:
                status, score, detail = "bullish", 1, "MACD 刚上穿信号线，形成金叉"
            elif prev_macd is not None and prev_signal is not None and prev_macd > prev_signal and macd <= macd_signal:
                status, score, detail = "bearish", -1, "MACD 刚下穿信号线，形成死叉"
            elif macd > macd_signal:
                status, score, detail = "bullish", 1, "MACD 位于信号线上方，动能偏多"
            elif macd < macd_signal:
                status, score, detail = "bearish", -1, "MACD 位于信号线下方，动能偏空"
            else:
                status, score, detail = "neutral", 0, "MACD 贴近信号线，方向不明"
            items.append({
                "key": "macd", "label": "MACD", "status": status, "score": score,
                "value": f"{macd:.3f} / {macd_signal:.3f}", "detail": detail,
            })

        rsi = number(latest, "RSI14")
        if rsi is not None:
            if rsi >= 70:
                status, score, detail = "risk", 0, "RSI 进入超买区，趋势强但追高风险上升"
            elif rsi <= 30:
                status, score, detail = "risk", 0, "RSI 进入超卖区，有反弹可能但不代表已止跌"
            elif rsi >= 55:
                status, score, detail = "bullish", 1, "RSI 处于强势区且尚未超买"
            elif rsi <= 45:
                status, score, detail = "bearish", -1, "RSI 处于弱势区且尚未超卖"
            else:
                status, score, detail = "neutral", 0, "RSI 位于多空平衡区"
            items.append({
                "key": "rsi", "label": "RSI(14)", "status": status, "score": score,
                "value": f"{rsi:.1f}", "detail": detail,
            })

        k = number(latest, "KDJ_K")
        d = number(latest, "KDJ_D")
        prev_k = number(previous, "KDJ_K")
        prev_d = number(previous, "KDJ_D")
        if k is not None and d is not None:
            if prev_k is not None and prev_d is not None and prev_k < prev_d and k >= d:
                status, score, detail = "bullish", 1, "K 线刚上穿 D 线，形成金叉"
            elif prev_k is not None and prev_d is not None and prev_k > prev_d and k <= d:
                status, score, detail = "bearish", -1, "K 线刚下穿 D 线，形成死叉"
            elif k >= 80 and d >= 80:
                status, score, detail = "risk", 0, "KDJ 位于高位超买区，注意钝化或回落"
            elif k <= 20 and d <= 20:
                status, score, detail = "risk", 0, "KDJ 位于低位超卖区，等待拐头确认"
            elif k > d:
                status, score, detail = "bullish", 1, "K 线位于 D 线上方，短线偏多"
            elif k < d:
                status, score, detail = "bearish", -1, "K 线位于 D 线下方，短线偏空"
            else:
                status, score, detail = "neutral", 0, "KDJ 方向不明"
            items.append({
                "key": "kdj", "label": "KDJ", "status": status, "score": score,
                "value": f"K {k:.1f} / D {d:.1f}", "detail": detail,
            })

        bb_middle = number(latest, "BB_middle")
        bb_upper = number(latest, "BB_upper")
        bb_lower = number(latest, "BB_lower")
        if price and bb_middle and bb_upper and bb_lower:
            if price >= bb_upper:
                status, score, detail = "risk", 0, "价格触及或突破布林上轨，强势但短线过热"
            elif price <= bb_lower:
                status, score, detail = "risk", 0, "价格触及或跌破布林下轨，弱势且可能超跌"
            elif price >= bb_middle:
                status, score, detail = "bullish", 1, "价格位于布林中轨上方，区间趋势偏多"
            else:
                status, score, detail = "bearish", -1, "价格位于布林中轨下方，区间趋势偏空"
            items.append({
                "key": "bollinger", "label": "布林带", "status": status, "score": score,
                "value": f"{price:.2f} / 中轨 {bb_middle:.2f}", "detail": detail,
            })

        volume = number(latest, "volume")
        volume_ma20 = number(latest, "VOL_MA20")
        pct_chg = number(latest, "pct_chg")
        if volume is not None and volume_ma20 and volume_ma20 > 0:
            ratio = volume / volume_ma20
            close = number(latest, "close")
            prior_window = df.iloc[-21:-1] if len(df) >= 21 else df.iloc[:-1]
            prior_high = None
            prior_low = None
            if not prior_window.empty:
                high_column = "high" if "high" in prior_window.columns else "close"
                low_column = "low" if "low" in prior_window.columns else "close"
                highs = pd.to_numeric(prior_window[high_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                lows = pd.to_numeric(prior_window[low_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not highs.empty:
                    prior_high = float(highs.max())
                if not lows.empty:
                    prior_low = float(lows.min())
            breakout = close is not None and prior_high is not None and close > prior_high
            breakdown = close is not None and prior_low is not None and close < prior_low

            if ratio >= 1.2 and pct_chg is not None and pct_chg > 0 and breakout:
                pattern, status, score = "放量突破", "bullish", 1
                detail = "收盘突破前20日高点且放量，属于突破特征；需后续不跌回原区间确认"
            elif ratio >= 1.2 and pct_chg is not None and pct_chg > 0:
                pattern, status, score = "放量上涨", "bullish", 1
                detail = "量价配合偏多，但尚未突破前20日高点，不按有效突破处理"
            elif ratio >= 1.2 and pct_chg is not None and pct_chg < 0 and breakdown:
                pattern, status, score = "放量破位", "bearish", -1
                detail = "收盘跌破前20日低点且放量，卖压确认偏强，不按洗盘处理"
            elif ratio >= 1.2 and pct_chg is not None and pct_chg < 0:
                pattern, status, score = "放量下跌", "bearish", -1
                detail = "具备派发或出货特征，但单日不能确认；需观察是否继续放量走弱"
            elif ratio <= 0.7 and pct_chg is not None and pct_chg < 0 and ma20 and close and close >= ma20 and not breakdown:
                pattern, status, score = "缩量回调", "neutral", 0
                detail = "仍在MA20上方且未破前低，具备洗盘特征但不能确认；需缩量止跌后重新放量上涨"
            elif ratio <= 0.7 and pct_chg is not None and pct_chg < 0:
                pattern, status, score = "缩量阴跌", "bearish", -1
                detail = "量能萎缩但趋势位置偏弱，不能仅按洗盘解释，等待止跌信号"
            elif ratio <= 0.7 and pct_chg is not None and pct_chg > 0:
                pattern, status, score = "缩量上涨", "risk", 0
                detail = "价格上涨但量能跟进不足，突破可靠性偏低"
            elif ratio >= 1.2:
                pattern, status, score = "放量分歧", "risk", 0
                detail = "成交量明显放大但价格方向不强，多空分歧加剧"
            elif ratio <= 0.7:
                pattern, status, score = "缩量整理", "neutral", 0
                detail = "成交量明显低于20日均量，方向尚未确认"
            else:
                pattern, status, score = "常量波动", "neutral", 0
                detail = "成交量接近20日常态水平，暂无显著量价信号"
            items.append({
                "key": "volume", "label": "量价形态", "status": status, "score": score,
                "value": f"{pattern} · {ratio:.2f}×20日均量", "detail": detail,
            })

        atr = number(latest, "ATR14")
        close = number(latest, "close")
        if atr is not None and close and close > 0:
            atr_pct = atr / close
            if atr_pct >= 0.04:
                status, detail = "risk", "ATR 占收盘价比例较高，日内波动风险偏大"
            else:
                status, detail = "neutral", "ATR 仅表示波动幅度，不单独判断涨跌方向"
            items.append({
                "key": "atr", "label": "ATR(14)", "status": status, "score": 0,
                "value": f"{atr:.2f} ({atr_pct:.1%})", "detail": detail,
            })

        if len(df) >= 21:
            close_20 = number(df.iloc[-21], "close")
            if close and close_20 and close_20 > 0:
                momentum = close / close_20 - 1
                if momentum > 0.03:
                    status, score, detail = "bullish", 1, "20日收益为正，中短期动量偏多"
                elif momentum < -0.03:
                    status, score, detail = "bearish", -1, "20日收益为负，中短期动量偏空"
                else:
                    status, score, detail = "neutral", 0, "20日涨跌幅较小，动量不明显"
                items.append({
                    "key": "momentum", "label": "20日动量", "status": status, "score": score,
                    "value": f"{momentum:+.1%}", "detail": detail,
                })

        bullish = sum(item["status"] == "bullish" for item in items)
        bearish = sum(item["status"] == "bearish" for item in items)
        neutral = sum(item["status"] == "neutral" for item in items)
        risk = sum(item["status"] == "risk" for item in items)
        total_score = sum(item["score"] for item in items)
        if total_score >= 2:
            bias, label = "bullish", "利好占优"
        elif total_score <= -2:
            bias, label = "bearish", "利空占优"
        else:
            bias, label = "neutral", "多空均衡"

        as_of = latest.get("trade_date") or latest.get("date") or latest.name
        if hasattr(as_of, "strftime"):
            as_of = as_of.strftime("%Y-%m-%d")
        else:
            as_of = str(as_of)[:10]
        return {
            "as_of": as_of,
            "price": round(price, 3) if price else None,
            "overall": {
                "bias": bias, "label": label, "score": total_score,
                "bullish": bullish, "bearish": bearish, "neutral": neutral, "risk": risk,
            },
            "items": items,
        }

    @staticmethod
    def detect_cross_signals(df: pd.DataFrame) -> list:
        """检测金叉/死叉等信号"""
        signals = []
        if len(df) < 2:
            return signals

        p, c = df.iloc[-2], df.iloc[-1]  # prev, curr

        # MACD 金叉/死叉
        try:
            if p.get("MACD", 0) < p.get("MACD_signal", 0) and c.get("MACD", 0) >= c.get("MACD_signal", 0):
                signals.append(("金叉", "MACD 金叉"))
            elif p.get("MACD", 0) > p.get("MACD_signal", 0) and c.get("MACD", 0) <= c.get("MACD_signal", 0):
                signals.append(("死叉", "MACD 死叉"))
        except Exception:
            pass

        # 均线金叉/死叉
        try:
            if p.get("MA5", 0) < p.get("MA20", 0) and c.get("MA5", 0) >= c.get("MA20", 0):
                signals.append(("金叉", "MA5 上穿 MA20"))
            elif p.get("MA5", 0) > p.get("MA20", 0) and c.get("MA5", 0) <= c.get("MA20", 0):
                signals.append(("死叉", "MA5 下穿 MA20"))
        except Exception:
            pass

        # KDJ 金叉/死叉
        try:
            if p.get("KDJ_K", 0) < p.get("KDJ_D", 0) and c.get("KDJ_K", 0) >= c.get("KDJ_D", 0):
                signals.append(("金叉", "KDJ 金叉"))
            elif p.get("KDJ_K", 0) > p.get("KDJ_D", 0) and c.get("KDJ_K", 0) <= c.get("KDJ_D", 0):
                signals.append(("死叉", "KDJ 死叉"))
        except Exception:
            pass

        return signals

    @staticmethod
    def detect_kline_patterns(df: pd.DataFrame) -> list:
        """检测 K 线形态"""
        patterns = []
        if len(df) < 2:
            return patterns

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None

        o, c_val, h, l = latest["open"], latest["close"], latest["high"], latest["low"]
        body = abs(c_val - o)
        upper_shadow = h - max(o, c_val)
        lower_shadow = min(o, c_val) - l
        total_range = h - l

        if total_range == 0:
            return patterns

        # 十字星
        if body / total_range < 0.1:
            patterns.append(("十字星", "多空力量平衡，变盘信号"))

        # 锤子线 / 吊颈线
        if prev is not None:
            prev_close = prev["close"]
            if lower_shadow > 2 * body and upper_shadow < body * 0.3:
                if c_val > prev_close or lower_shadow > total_range * 0.6:
                    patterns.append(("锤子线", "下影线长，可能见底反转"))
                else:
                    patterns.append(("吊颈线", "高位长下影，可能见顶"))

            # 吞没形态
            prev_body = abs(prev["close"] - prev["open"])
            if body > prev_body * 1.2:
                if c_val > o and prev["close"] < prev["open"] and c_val > prev["open"] and o < prev["close"]:
                    patterns.append(("阳包阴", "看涨吞没形态"))
                elif c_val < o and prev["close"] > prev["open"] and c_val < prev["open"] and o > prev["close"]:
                    patterns.append(("阴包阳", "看跌吞没形态"))

        # 三连阳 / 三连阴
        if len(df) >= 4:
            last3 = df.iloc[-4:]
            if all(r["close"] > r["open"] for _, r in last3.iterrows()):
                patterns.append(("三连阳", "连续三天收阳，强势信号"))
            elif all(r["close"] < r["open"] for _, r in last3.iterrows()):
                patterns.append(("三连阴", "连续三天收阴，弱势信号"))

        return patterns

    @staticmethod
    def get_analysis_context(df: pd.DataFrame, financial: Optional[dict] = None, indicators_added: bool = False) -> str:
        """生成供 LLM 分析的上下文文本"""
        if df.empty:
            return "数据不足"

        if not indicators_added:
            df = Analyzer.add_indicators(df)
        ind = Analyzer.latest_indicators(df)
        signals = Analyzer.detect_cross_signals(df)
        patterns = Analyzer.detect_kline_patterns(df)

        # 近期表现
        recent = df.tail(60)
        pct_5d = (recent["close"].iloc[-1] / recent["close"].iloc[-5] - 1) * 100 if len(recent) >= 5 else 0
        pct_20d = (recent["close"].iloc[-1] / recent["close"].iloc[-20] - 1) * 100 if len(recent) >= 20 else 0
        pct_60d = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100 if len(recent) >= 60 else 0

        lines = [
            "=== 近期表现 ===",
            f"5日涨跌幅: {pct_5d:.2f}%",
            f"20日涨跌幅: {pct_20d:.2f}%",
            f"60日涨跌幅: {pct_60d:.2f}%",
            "",
            "=== 技术指标 ===",
        ]
        for k, v in ind.items():
            lines.append(f"{k}: {v}")

        if signals:
            lines.extend(["", "=== 交叉信号 ==="])
            for sig_type, desc in signals:
                emoji = "🟢" if sig_type == "金叉" else "🔴"
                lines.append(f"{emoji} {desc}")

        if patterns:
            lines.extend(["", "=== K线形态 ==="])
            for p_name, p_desc in patterns:
                lines.append(f"• {p_name}: {p_desc}")

        if financial:
            lines.extend(["", "=== 财务摘要 ==="])
            for k, v in financial.items():
                lines.append(f"{k}: {v}")

        return "\n".join(lines)
