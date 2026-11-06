"""情绪分类器——本地GPU模型(主力) + DeepSeek LLM(交叉验证) + 关键词(兜底)"""
import json
import logging
import os
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 本地 GPU 模型（惰性加载）──
_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_inference_lock = threading.Lock()
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            _pipeline = pipeline(
                "text-classification",
                model=MODEL_NAME,
                device=device,
                max_length=128,
                truncation=True,
            )
            logger.info(f"[本地模型] {MODEL_NAME} 已加载 (device={'GPU' if device == 0 else 'CPU'})")
            return _pipeline
        except Exception as e:
            logger.warning(f"本地模型加载失败: {e}，回退到LLM/关键词")
            return None


def _normalize_local_result(result):
    label = str(result["label"]).lower()
    score = float(result["score"])
    if "positive" in label or "pos" in label[:4] or "正面" in label:
        value = (score - 0.5) * 2.0
        return {"sentiment": "positive", "score": round(max(0, value), 3),
                "confidence": round(score, 3)}
    if "negative" in label or "neg" in label[:4] or "负面" in label:
        value = (score - 0.5) * 2.0
        return {"sentiment": "negative", "score": round(min(0, -value), 3),
                "confidence": round(score, 3)}
    return {"sentiment": "neutral", "score": 0.0,
            "confidence": round(1.0 - score, 3)}


def _local_sentiments(titles, batch_size=64):
    try:
        pipe = _get_pipeline()
        if pipe is None:
            return {}
        with _pipeline_inference_lock:
            outputs = pipe(
                titles,
                truncation=True,
                max_length=128,
                batch_size=min(batch_size, len(titles)),
            )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return {
            title: _normalize_local_result(result)
            for title, result in zip(titles, outputs)
        }
    except Exception as e:
        logger.debug(f"本地模型推理失败: {e}")
        return {}


def _local_sentiment(title):
    return _local_sentiments([title], batch_size=1).get(title)


# ── 关键词词典（终极兜底）──
POSITIVE_WORDS = [
    "增长", "大涨", "涨停", "突破", "创新高", "超预期", "扭亏", "盈利", "分红",
    "回购", "增持", "中标", "签约", "获批", "上市", "利好", "预增", "预盈",
    "订单", "扩产", "新品", "专利", "战略合作", "业绩", "提升", "加速",
    "买入", "推荐", "看好", "增资", "升级", "摘帽", "恢复", "复苏",
]
NEGATIVE_WORDS = [
    "下跌", "大跌", "跌停", "亏损", "预亏", "预减", "减持", "退市", "ST",
    "违规", "处罚", "立案", "调查", "诉讼", "债务", "违约", "爆雷", "暴跌",
    "下滑", "下降", "萎缩", "停工", "停产", "裁员", "关闭", "终止",
    "冻结", "质押", "平仓", "强平", "清仓", "卖出", "看空", "降级",
    "减值", "计提", "商誉", "雷", "风险", "警示", "暂停", "问询",
]


def keyword_sentiment(title):
    pos = sum(1 for w in POSITIVE_WORDS if w in title)
    neg = sum(1 for w in NEGATIVE_WORDS if w in title)
    if pos > neg:
        return {"sentiment": "positive", "score": min(0.8, 0.3 + pos * 0.15),
                "confidence": 0.35 + min(0.2, (pos - neg) * 0.1)}
    elif neg > pos:
        return {"sentiment": "negative", "score": max(-0.8, -0.3 - neg * 0.15),
                "confidence": 0.35 + min(0.2, (neg - pos) * 0.1)}
    return {"sentiment": "neutral", "score": 0, "confidence": 0.25}


class SentimentAnalyzer:
    def __init__(self):
        self.batch_size = 64

    def analyze(self, items):
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            results.extend(self._analyze_batch(batch))
        return results

    def _analyze_batch(self, items):
        titles = [it["title"] for it in items if it.get("title")]
        if not titles:
            return items

        # 1. 本地 GPU 模型
        local_scores = _local_sentiments(titles, self.batch_size)
        for t, r in local_scores.items():
            kw = keyword_sentiment(t)
            if kw["sentiment"] != "neutral" and r["sentiment"] != "neutral":
                if kw["sentiment"] != r["sentiment"]:
                    r["confidence"] = round(r["confidence"] * 0.5, 3)

        # 2. 本地模型全失败 → LLM
        if not local_scores:
            local_scores = self._try_llm(titles)

        for item in items:
            t = item.get("title", "")
            source_conf = float(item.get("source_confidence", 0.50))

            if t in local_scores:
                s = local_scores[t]
                model_conf = float(s.get("confidence", 0.50))
                combined_conf = round(model_conf * 0.7 + source_conf * 0.3, 3)
                item["sentiment"] = s.get("sentiment", "neutral")
                item["sentiment_score"] = s.get("score", 0)
                item["llm_confidence"] = model_conf
                item["source_confidence"] = source_conf
                item["confidence"] = combined_conf
            else:
                s = keyword_sentiment(t)
                kw_conf = float(s.get("confidence", 0.25))
                combined_conf = round(kw_conf * 0.6 + source_conf * 0.4, 3)
                item["sentiment"] = s.get("sentiment", "neutral")
                item["sentiment_score"] = s.get("score", 0)
                item["llm_confidence"] = kw_conf
                item["source_confidence"] = source_conf
                item["confidence"] = combined_conf

            item["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return items

    def _try_llm(self, titles):
        """DeepSeek LLM 回退"""
        try:
            from core.shared.infrastructure.llm_client import LLMClient
            llm = LLMClient()
            prompt = self._build_prompt(titles)
            resp = llm.chat([
                {"role": "system", "content": (
                    "你是一个金融情绪分析专家。对每条新闻标题输出:\n"
                    "- sentiment: positive(利好)/negative(利空)/neutral(中性)\n"
                    "- score: -1.0(强利空) 到 +1.0(强利好)\n"
                    "- confidence: 0-1.0 (0.9+=明确, 0.5-0.7=有指向但信息不足, <0.3=纯噪音)\n"
                    "返回纯JSON数组。"
                )},
                {"role": "user", "content": prompt},
            ])
            return self._parse_llm_response(resp, titles)
        except Exception as e:
            logger.debug(f"LLM情绪分析不可用: {e}")
            return {}

    def _build_prompt(self, titles):
        lines = [f"{i + 1}. {t}" for i, t in enumerate(titles)]
        return "分析以下新闻标题的情绪倾向，返回JSON数组:\n" + "\n".join(lines)

    def _parse_llm_response(self, resp, titles):
        try:
            text = resp if isinstance(resp, str) else (resp.get("content", "") if isinstance(resp, dict) else str(resp))
            if "[" in text and "]" in text:
                text = text[text.index("["):text.rindex("]") + 1]
            arr = json.loads(text)
            result = {}
            for item in arr:
                t = item.get("title", "")
                if t:
                    result[t] = {
                        "sentiment": item.get("sentiment", "neutral"),
                        "score": item.get("score", 0),
                        "confidence": item.get("confidence", 0.5),
                    }
            return result
        except Exception:
            result = {}
            for i, t in enumerate(titles):
                try:
                    parsed = json.loads(text) if i == 0 else parsed
                    item = parsed[i] if isinstance(parsed, list) else {}
                    result[t] = {
                        "sentiment": item.get("sentiment", "neutral"),
                        "score": item.get("score", 0),
                        "confidence": item.get("confidence", 0.5),
                    }
                except Exception:
                    pass
            return result
