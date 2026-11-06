"""持仓管理 + 风险扫描引擎"""
import logging
from typing import List, Dict, Optional
 
from core.shared import config
from core.domain.screening.indicators import Analyzer
from core.shared.infrastructure.cache import CacheManager
from core.shared.infrastructure.llm import LLMClient
 
logger = logging.getLogger(__name__)


def _check_tech_red_flags(df, sig) -> list:
    """用技术指标做确定性判定。返回触发的恶化信号列表，空列表=无恶化。
    这些检查不依赖LLM，直接从DataFrame提取数值，零假阴性风险。
    """
    flags = []
    if df is None or df.empty or len(df) < 20:
        return flags

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    def _f(val, default=0.0):
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    close = _f(last.get("close"))
    prev_close = _f(prev.get("close"))
    volume = _f(last.get("volume"))
    vol_ma20 = _f(last.get("VOL_MA20"))

    # 1. 交叉信号直接判定
    if sig:
        sig_text = " ".join(f"{t}:{d}" for t, d in sig)
        if "死叉" in sig_text:
            flags.append("死叉信号")
        if "顶背离" in sig_text:
            flags.append("顶背离")

    # 2. 跌破关键均线 (当日收盘价低于MA20或MA60)
    ma20 = _f(last.get("MA20"))
    ma60 = _f(last.get("MA60"))
    if ma20 > 0 and 0 < close < ma20:
        flags.append(f"跌破MA20(close={close:.1f})")
    if ma60 > 0 and 0 < close < ma60:
        flags.append(f"跌破MA60(close={close:.1f})")

    # 3. RSI极端值
    rsi = _f(last.get("RSI14"), 50)
    if rsi > 80:
        flags.append(f"RSI超买({rsi:.0f})")
    elif rsi < 20:
        flags.append(f"RSI超卖({rsi:.0f})")

    # 4. 放量下跌 (量>1.5倍均量 且 跌幅>2%)
    if vol_ma20 > 0 and volume > 1.5 * vol_ma20:
        pct = (close - prev_close) / prev_close if prev_close > 0 else 0
        if pct < -0.02:
            flags.append(f"放量下跌(量{volume/vol_ma20:.1f}倍, 跌{pct*100:.1f}%)")

    # 5. 均线空头排列 MA5 < MA10 < MA20 < MA60
    ma5 = _f(last.get("MA5"))
    ma10 = _f(last.get("MA10"))
    if all(v > 0 for v in [ma5, ma10, ma20, ma60]) and ma5 < ma10 < ma20 < ma60:
        flags.append("均线空头排列")

    return flags


class PortfolioScanner:
    """扫描持仓, 逐个进行多Agent辩论 + 风险评分"""
 
    def __init__(self, debate_panel=None, fetcher=None, analyzer=None):
        from core.data.fetcher import DataFetcher
        from core.data.sources.realtime import RealtimeEngine
        from agents.panel import DebatePanel
        self.cache = CacheManager()
        self.fetcher = fetcher or DataFetcher(self.cache)
        self.analyzer = analyzer or Analyzer()
        self.realtime = RealtimeEngine()
        llm = LLMClient()
        self.panel = debate_panel or DebatePanel(llm=llm, cache=self.cache)
 
    def scan_holding(self, holding: Dict, include: Optional[Dict[str, bool]] = None) -> Optional[Dict]:
        """扫描单只持仓，返回风险评估结果"""
        inc = include or {}
        code = holding["code"]
        name = holding["name"]
        shares = holding["shares"]
        cost_price = float(holding["cost_price"])

        logger.info(f"扫描持仓: {name}({code}) inc={inc}")
        enabled = []
        skipped = []

        if not inc.get("daily", True):
            raise ValueError("日K线不能禁用")
        try:
            df = self.fetcher.get_stock_daily(code)
        except Exception as e:
            raise ValueError(f"日K线获取失败: {e}")
        if df is None or df.empty:
            raise ValueError("日K线数据为空")
        # MySQL Decimal → float（避免 add_indicators 报 Decimal/float 运算错误）
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        enabled.append("日K线")

        sig = []
        if inc.get("technical", True):
            try:
                df = self.analyzer.add_indicators(df)
                sig = self.analyzer.detect_cross_signals(df) or []
            except Exception as e:
                raise ValueError(f"技术指标计算失败: {e}")
            enabled.append("技术指标")
        else:
            skipped.append("技术指标")

        financial = None
        if inc.get("financial", True):
            try:
                financial = self.fetcher.get_stock_financial(code)
            except Exception as e:
                logger.warning(f"财务指标获取异常: {e}")
                financial = None
            if not financial:
                logger.warning(f"财务指标数据为空，跳过基本面分析")
                skipped.append("财务指标(无数据)")
            else:
                enabled.append("财务指标")
        else:
            skipped.append("财务指标")

        quote = None
        if inc.get("realtime", True):
            quote = self.realtime.get_realtime_price(code)
            if quote:
                from datetime import datetime
                from core.domain.market_data.freshness import assess_quote
                from core.shared.data_contract import data_envelope_allows_use, make_live_quote_envelope
                from core.shared.persistence import now_iso
                status = assess_quote(
                    code,
                    quote,
                    datetime.fromisoformat(now_iso()),
                    config.LIVE_QUOTE_MAX_AGE,
                )
                quote_envelope = make_live_quote_envelope(quote)
                if not status["usable"] or not data_envelope_allows_use(quote_envelope, "risk"):
                    quote = None
            if not quote:
                raise ValueError("实时行情不可用或数据契约未验证")
            enabled.append("实时行情")
        else:
            skipped.append("实时行情")

        patterns = []
        if inc.get("patterns", True):
            try:
                pat = self.analyzer.detect_kline_patterns(df)
                patterns = pat or []
            except Exception as e:
                raise ValueError(f"K线形态识别失败: {e}")
            enabled.append("K线形态")
        else:
            skipped.append("K线形态")

        data_context = self.analyzer.get_analysis_context(df, financial or {}, indicators_added=True)
        data_context += f"\n\n股票名称: {name}\n股票代码: {code}"
        data_context += f"\n启用数据源: {', '.join(enabled)}"
        if skipped:
            data_context += f"\n未启用数据源: {', '.join(skipped)}"
        if quote:
            pct = quote.get("pct_chg", quote.get("涨跌幅", 0))
            price = quote.get("price", quote.get("最新价", 0))
            data_context += f"\n实时价格: {price}\n实时涨跌幅: {pct}%\n"
        if sig:
            data_context += "\n技术交叉信号:\n"
            for t, d in sig:
                data_context += f"- {t}: {d}\n"
        if patterns:
            data_context += "\nK线形态:\n"
            for pn, pd in patterns[:5]:
                data_context += f"- {pn}: {pd}\n"

        # ── 确定性技术判定：指标恶化直接触发辩论，不依赖LLM ──
        tech_red_flags = _check_tech_red_flags(df, sig)
        if tech_red_flags:
            logger.info(f"[门控] {name}({code}) 技术指标恶化({', '.join(tech_red_flags)})，强制触发深度辩论")

        # ── 冲突门控：前置快速扫描 ──
        skip_debate = False
        if not tech_red_flags:
            try:
                from agents.prompts import PRE_SCREEN_PROMPT
                screener = LLMClient()
                screen_resp = screener.chat([
                    {"role": "system", "content": PRE_SCREEN_PROMPT},
                    {"role": "user", "content": data_context},
                ], temperature=0.2)
                screen_text = screen_resp.strip()
                if screen_text.startswith("不触发"):
                    skip_debate = True
                    logger.info(f"[门控] {name}({code}) 快速扫描通过，跳过深度辩论")
                else:
                    logger.info(f"[门控] {name}({code}) 触发辩论: {screen_text[:80]}")
            except Exception as e:
                logger.warning(f"[门控] 前置扫描失败，回退完整辩论: {e}")

        if skip_debate:
            final = f"快速筛查：正常持有\n风险评分: 2/10\n风险等级: 低风险\n机会评分: 5/10\n综合建议: 持仓信号正常，未发现需要警惕的风险因素，建议继续持有监控。"
            risk_level, risk_score = "低风险", 2
            debate_result = {"最终决议": final, "门控": "快速扫描通过，跳过深度辩论"}
        else:
            try:
                debate_result = self.panel.debate(name, code, data_context)
                final = debate_result.get("最终决议", "")
            except Exception as e:
                logger.error(f"辩论失败: {e}")
                return None

            # 从最终决议中提取风险等级和评分
            risk_level, risk_score = self._parse_risk(final)

        current_price = float(quote.get("price", quote.get("最新价", 0))) if quote else 0
        profit_loss = 0
        profit_loss_pct = 0
        if current_price and cost_price:
            profit_loss = (current_price - cost_price) * shares
            profit_loss_pct = (current_price - cost_price) / cost_price * 100
 
        # ── MetaSignal 持仓侧融合 ──
        fusion_result = None
        try:
            from core.domain.signals.meta import MetaSignalFusion
            fusion = MetaSignalFusion(mode="hold_side")

            # 1. 技术恶化检测
            tech_deterioration = 0.0
            if sig:
                sell_signals = sum(1 for _, d in sig if "死叉" in d or "跌破" in d or "顶背离" in d)
                buy_signals = sum(1 for _, d in sig if "金叉" in d or "突破" in d or "底背离" in d)
                if sell_signals + buy_signals > 0:
                    tech_deterioration = -(sell_signals - buy_signals) / max(1, sell_signals + buy_signals)
            fusion.add("tech_fusion", tech_deterioration, 0.8)

            # 2. Qlib 预测
            qlib_score = 0.0
            try:
                from core.domain.screening._screener import get_screener
                screener = get_screener()
                if screener.qlib_predictions and code in screener.qlib_predictions:
                    pred_val = screener.qlib_predictions[code]
                    preds = list(screener.qlib_predictions.values())
                    qlib_score = MetaSignalFusion.qlib_to_score(pred_val, preds)
            except Exception:
                pass
            fusion.add("qlib", qlib_score, getattr(config, "QLIB_DEFAULT_IC", 0.5))

            # 3. 情绪
            sentiment_score = 0.0
            try:
                from core.data import Database
                conn = Database.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT AVG(avg_score) as s FROM sentiment_daily "
                        "WHERE code=%s AND trade_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)",
                        (code,),
                    )
                    row = cur.fetchone()
                    cur.close()
                    if row and row["s"] is not None:
                        sentiment_score = max(-1.0, min(1.0, float(row["s"])))
                finally:
                    Database.return_connection(conn)
            except Exception:
                pass
            fusion.add("sentiment", sentiment_score, 0.5)

            # 5. 盈亏压力
            pnl_score = MetaSignalFusion.calc_pnl_pressure(
                cost_price, current_price, shares,
                total_asset=0, hold_days=0
            )
            fusion.add("pnl_pressure", pnl_score, 0.9)

            # 6. AI 辩论叠加
            if risk_score is not None:
                opportunity_score = 5
                import re
                om = re.search(r"机会评分:\s*(\d+)/10", final)
                if om:
                    opportunity_score = int(om.group(1))
                debate_norm = (opportunity_score - risk_score) / 10
                fusion.add("ai_debate", debate_norm, 0.85)

            fused = fusion.fuse()
            fusion_result = {
                "score": fused.score,
                "direction": fused.direction,
                "confidence": fused.confidence,
                "action": fused.action,
                "size_ratio": fused.size_ratio,
                "coverage": fused.coverage,
                "reason": fused.reason,
                "breakdown": {
                    v.source: {
                        "score": round(v.score, 2),
                        "direction": "bullish" if v.direction > 0 else "bearish" if v.direction < 0 else "neutral",
                    }
                    for v in fused.votes
                },
            }
        except Exception as e:
            logger.warning(f"MetaSignal融合失败: {e}")

        result = {
            "stock_code": code,
            "stock_name": name,
            "shares": shares,
            "cost_price": cost_price,
            "current_price": current_price,
            "profit_loss": round(profit_loss, 2),
            "profit_loss_pct": round(profit_loss_pct, 2),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_status": "available" if risk_score is not None else "unknown",
            "risk_detail": final,
            "suggestion": self._extract_suggestion(final),
            "debate_result": debate_result,
            "fusion": fusion_result,
            "data_enabled": enabled,
            "data_skipped": skipped,
        }
        return result
 
    def scan_all(self, threshold: int = 5, include: Optional[Dict[str, bool]] = None) -> List[Dict]:
        """扫描所有启用持仓并返回本轮逐项状态"""
        import uuid
        from core.data import HoldingsRepo
        holdings = HoldingsRepo.get_all()
        scan_id = str(uuid.uuid4())
        if not holdings:
            logger.info("无持仓记录")
            return []
 
        alerts = []
        for h in holdings:
            code = str(h.get("code") or "")
            base_result = {
                "stock_code": code,
                "stock_name": h.get("name", ""),
                "risk_level": "未知",
                "risk_score": None,
                "risk_status": "unknown",
                "scan_id": scan_id,
                "alert_triggered": False,
            }
            if not h.get("alerts_enabled", True):
                base_result["risk_status"] = "disabled"
                alerts.append(base_result)
                continue
            try:
                result = self.scan_holding(h, include=include)
                if not result:
                    base_result["risk_status"] = "failed"
                    base_result["failure_code"] = "empty_result"
                    alerts.append(base_result)
                    continue
                result["scan_id"] = scan_id
                result.setdefault("risk_status", "available")
                result.setdefault("alert_triggered", False)
                if result.get("risk_score") is None:
                    result["risk_status"] = "unknown"
                    alerts.append(result)
                else:
                    result["alert_triggered"] = result["risk_score"] >= threshold
                    alerts.append(result)
                    if result["alert_triggered"]:
                        notify_threshold = h.get("risk_threshold", 7)
                        if notify_threshold is None:
                            notify_threshold = 7
                        notify = result["risk_score"] >= notify_threshold
                        alert_id = self.persist_alert(result, notify=notify)
                        result["alert_id"] = alert_id
                        result["notification_status"] = (
                            "pending" if notify and alert_id is not None
                            else "unavailable" if notify
                            else "not_required"
                        )
            except Exception as e:
                logger.error(f"扫描 {h.get('code','?')} 失败: {e}")
                base_result["risk_status"] = "failed"
                base_result["failure_code"] = type(e).__name__
                alerts.append(base_result)
        return alerts
 
    def _parse_risk(self, text: str):
        """从最终决议中解析风险等级和评分"""
        import re
        if not isinstance(text, str) or not text.strip():
            return "未知", None

        # 优先匹配固定格式区块
        block = re.search(r"风险评分:\s*(?<![\d.+-])(\d{1,2})/10\s*\n\s*风险等级:\s*(高风险|中风险|低风险)", text)
        if block:
            score = int(block.group(1))
            level = block.group(2)
            if not 0 <= score <= 10:
                return "未知", None
            return level, score

        # 回退匹配自由格式
        m = re.search(r"风险评分[：:]\s*(?<![\d.+-])(\d{1,2})(?:/10)?(?![\d.])", text)
        if m:
            score = int(m.group(1))
            if not 0 <= score <= 10:
                return "未知", None
            if score >= 7:
                level = "高风险"
            elif score >= 5:
                level = "中风险"
            else:
                level = "低风险"
            return level, score

        if "高风险" in text:
            return "高风险", 7
        if "中风险" in text:
            return "中风险", 5
        return "未知", None
 
    def _extract_suggestion(self, text: str) -> str:
        """从最终决议中提取建议部分"""
        import re
        m = re.search(r"(?:建议|操作建议|投资建议)[：:](.*?)(?:\n\n|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""
 
    def persist_alert(self, result: Dict, notify=False):
        """保存告警记录到数据库"""
        if result.get("risk_score") is None:
            return None
        from uuid import uuid4
        from core.data import Database, RiskAlertRepo
        if not Database.is_available():
            return None
        notification_key = None
        notification_payload = None
        if notify:
            scan_id = result.get("scan_id") or str(uuid4())
            notification_key = f"ai-risk:{scan_id}:{result['stock_code']}"
            notification_payload = {
                "stock_name": result["stock_name"],
                "stock_code": result["stock_code"],
                "risk_level": result["risk_level"],
                "risk_score": result["risk_score"],
                "risk_detail": result.get("risk_detail", "")[:1000],
                "suggestion": result.get("suggestion", "")[:500],
                "shares": result.get("shares", 0),
                "cost_price": result.get("cost_price", 0),
            }
        return RiskAlertRepo.add(
            stock_code=result["stock_code"],
            stock_name=result["stock_name"],
            risk_level=result["risk_level"],
            risk_score=result["risk_score"],
            risk_detail=result.get("risk_detail", "")[:2000],
            suggestion=result.get("suggestion", "")[:500],
            notification_payload=notification_payload,
            idempotency_key=notification_key,
        )
