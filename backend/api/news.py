"""新闻数据 API"""
import anyio
from fastapi import APIRouter
router = APIRouter()

@router.get("/{code}")
async def get_stock_news(code: str, limit: int = 50):
    from core.adapters.nlp.collector import SentimentCollector
    news = await anyio.to_thread.run_sync(SentimentCollector().get_latest_news, code, limit)
    return {"data": news}

@router.get("/{code}/refresh")
async def refresh_news(code: str):
    def _run():
        from core.adapters.nlp.collector import SentimentCollector
        from core.adapters.nlp.analyzer import SentimentAnalyzer
        c = SentimentCollector(); a = SentimentAnalyzer()
        items = c.dedup(c.collect_single(code))
        analyzed = a.analyze(items)
        c.save_to_db(analyzed)
        return {"count": len(analyzed), "code": code}
    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}

@router.get("/latest/list")
async def get_latest_news_all(limit: int = 20):
    from core.adapters.nlp.collector import SentimentCollector
    news = await anyio.to_thread.run_sync(SentimentCollector().get_all_latest)
    return {"data": news[:limit]}
