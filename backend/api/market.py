"""大盘数据 API"""
import anyio
from fastapi import APIRouter
from core.data.fetcher import DataFetcher
from core.data.sources.realtime import RealtimeEngine
from backend.service.market_data import MarketDataService

router = APIRouter()
fetcher = DataFetcher()
rt = RealtimeEngine()


@router.get("/indices")
async def get_indices():
    """四大指数实时行情"""
    codes = ["sh000001", "sz399001", "sz399006", "sh000688"]
    return await anyio.to_thread.run_sync(MarketDataService.get_live_quotes, codes, "live_display", "index")


@router.get("/index-chart")
async def get_index_chart(index_name: str = "上证指数", days: int = 120):
    """指数历史走势"""
    try:
        df = await anyio.to_thread.run_sync(fetcher.get_index_daily, index_name, days)
        if df is None or df.empty:
            return {"data": []}
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        return {"data": [], "error": str(e)}


@router.get("/top-gainers")
async def top_gainers(n: int = 20):
    """涨幅榜"""
    return await anyio.to_thread.run_sync(MarketDataService.get_live_ranking, False, n)


@router.get("/top-losers")
async def top_losers(n: int = 20):
    """跌幅榜"""
    return await anyio.to_thread.run_sync(MarketDataService.get_live_ranking, True, n)


@router.get("/sectors")
async def get_sectors():
    """板块热点"""
    return await anyio.to_thread.run_sync(MarketDataService.get_live_sectors)
