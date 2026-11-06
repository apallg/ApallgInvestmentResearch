"""情绪因子计算器——置信度加权 + 多源交叉验证"""
from datetime import datetime

import numpy as np


class SentimentFactorCalculator:
    def __init__(self):
        self.windows = [3, 5, 10]
        self._min_confidence = 0.35  # 低于此置信度的新闻不计入因子

    def _merge_notices(self, code, items):
        """从 stock_notices 表读取当日公告，转为与新闻一致的格式合并"""
        try:
            from core.data import query_sql
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            notices = query_sql(
                "SELECT title, notice_type, source, source_confidence FROM stock_notices "
                "WHERE code=%s AND notice_date=%s",
                (code, today))
            if not notices:
                return items
            for n in notices:
                title = n.get("title", "")
                ntype = n.get("notice_type", "其他")
                source_conf = float(n.get("source_confidence", 0.95))
                score = 0.0
                if ntype in ("增减持", "风险提示"):
                    score = -0.4
                elif ntype == "业绩预告":
                    if any(w in title for w in ["预增", "预盈", "扭亏", "增长"]):
                        score = 0.5
                    elif any(w in title for w in ["预减", "预亏", "下降"]):
                        score = -0.5
                elif ntype == "重大事项":
                    if any(w in title for w in ["重组", "收购"]):
                        score = 0.3
                    elif any(w in title for w in ["停牌", "终止"]):
                        score = -0.3
                items.append({
                    "title": title,
                    "code": code,
                    "date": today,
                    "sentiment": "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral",
                    "sentiment_score": score,
                    "confidence": source_conf,
                    "source_tag": "notice",
                    "source_confidence": source_conf,
                })
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"merge_notices({code}): {e}")
        return items

    def calc_daily(self, code, analyzed_items):
        """计算单只股票每日情绪因子。

        三层置信度已在前序步骤处理:
        1. 来源置信度 (collector: 信源类型预设)
        2. LLM把握度 (analyzer: LLM自评0-1)
        3. 交叉验证 (此处: 多源同向则提升)

        Returns None if insufficient data.
        """
        if analyzed_items is None:
            analyzed_items = []
        analyzed_items = self._merge_notices(code, analyzed_items)

        if not analyzed_items:
            return None

        # ── 仅保留当日新闻 ──
        today = datetime.now().strftime("%Y-%m-%d")
        analyzed_items = [it for it in analyzed_items if it.get("date", "")[:10] == today]
        if not analyzed_items:
            return None

        # ── 过滤低置信度 ──
        valid = [it for it in analyzed_items
                 if it.get("sentiment_score") is not None
                 and float(it.get("confidence", 0)) >= self._min_confidence]

        if not valid:
            return None

        n = len(valid)
        scores = [float(it["sentiment_score"]) for it in valid]
        confidences = [float(it.get("confidence", 0.5)) for it in valid]

        # ── 置信度加权均分 ──
        total_conf = sum(confidences)
        if total_conf > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_conf
        else:
            weighted_score = sum(scores) / n

        # ── 多源交叉验证 ──
        # 1. 方向一致性
        directions = [1 if s > 0.05 else -1 if s < -0.05 else 0 for s in scores]
        aligned = sum(1 for d in directions if d != 0)
        # 2. 多源计数（不同 source_tag 的数量）
        source_tags = set(it.get("source_tag", "") for it in valid if it.get("source_tag"))
        multi_source = len(source_tags) >= 2

        if aligned >= 2:
            same_dir = max(
                sum(1 for d in directions if d > 0),
                sum(1 for d in directions if d < 0),
            )
            base_boost = min(0.25, (same_dir / n - 0.5) * 0.6)
            # 多源同向额外加成
            multi_source_bonus = 0.10 if multi_source else 0.0
            cross_val_boost = base_boost + multi_source_bonus
        elif aligned == 1 and multi_source:
            cross_val_boost = 0.05  # 单条但有多个来源交叉印证
        else:
            cross_val_boost = -0.15  # 仅1源1条 → 惩罚

        # ── 最终置信度 = 平均LLM/来源置信度 + 交叉验证加成 ──
        avg_raw_confidence = sum(confidences) / n
        final_confidence = max(0.0, min(1.0, avg_raw_confidence + cross_val_boost))

        # ── 低置信度过滤 ──
        if final_confidence < 0.35:
            return None

        pos = sum(1 for s in scores if s > 0.05)
        neg = sum(1 for s in scores if s < -0.05)

        return {
            "code": code,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "avg_score": round(weighted_score, 4),
            "pos_ratio": round(pos / n, 4),
            "neg_ratio": round(neg / n, 4),
            "news_count": n,
            "source_count": len(source_tags),
            "avg_confidence": round(final_confidence, 4),
            "cross_val_boost": round(cross_val_boost, 3),
        }

    def save_daily(self, daily):
        try:
            from core.data import execute_sql
            execute_sql(
                "INSERT INTO sentiment_daily "
                "(code,trade_date,avg_score,pos_ratio,neg_ratio,news_count,avg_confidence) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE "
                "avg_score=VALUES(avg_score),pos_ratio=VALUES(pos_ratio),"
                "neg_ratio=VALUES(neg_ratio),news_count=VALUES(news_count),"
                "avg_confidence=VALUES(avg_confidence)",
                (daily["code"], daily["date"], daily["avg_score"],
                 daily["pos_ratio"], daily["neg_ratio"], daily["news_count"],
                 daily.get("avg_confidence", 0.5)),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"save_daily: {e}")

    def get_history(self, code, days=30):
        try:
            from core.data import query_sql
            return query_sql(
                "SELECT * FROM sentiment_daily WHERE code=%s "
                "ORDER BY trade_date DESC LIMIT %s", (code, days)) or []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"get_history: {e}")
            return []
