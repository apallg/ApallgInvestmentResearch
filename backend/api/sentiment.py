"""情绪数据 API"""
from datetime import date, datetime, timedelta

import anyio
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/{code}")
async def get_stock_sentiment(code: str, days: int = 30):
    from core.adapters.nlp.factor import SentimentFactorCalculator
    data = await anyio.to_thread.run_sync(SentimentFactorCalculator().get_history, code, days)
    return {"data": data}


@router.post("/{code}/refresh")
async def refresh_stock_sentiment(code: str):
    def _run():
        from core.adapters.nlp.collector import SentimentCollector
        from core.adapters.nlp.analyzer import SentimentAnalyzer
        from core.adapters.nlp.factor import SentimentFactorCalculator
        collector = SentimentCollector()
        analyzer = SentimentAnalyzer()
        calculator = SentimentFactorCalculator()
        items = collector.dedup(collector.collect_single(code))
        if not items:
            return {"code": code, "count": 0, "message": "无新新闻"}
        analyzed = analyzer.analyze(items)
        collector.save_to_db(analyzed)
        daily = calculator.calc_daily(code, analyzed)
        if daily:
            calculator.save_daily(daily)
        return {"code": code, "count": len(analyzed), "daily": daily}
    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}


@router.get("/market/overview")
async def get_market_sentiment(days: int = 30):
    def _run():
        from core.data import Database
        if not Database.is_available():
            return {"today": {}, "trend": [], "news": []}
        today = datetime.now().strftime("%Y-%m-%d")
        trend_start = (datetime.now() - timedelta(days=max(days, 7))).strftime("%Y-%m-%d")
        td = Database.fetchall(
            "SELECT AVG(avg_score) as avg_score, SUM(news_count) as total_news, "
            "AVG(pos_ratio) as pos_ratio, AVG(neg_ratio) as neg_ratio "
            "FROM sentiment_daily WHERE trade_date=(SELECT MAX(trade_date) FROM sentiment_daily WHERE trade_date<=%s)",
            (today,))
        trend = Database.fetchall(
            "SELECT trade_date as date, AVG(avg_score) as score, SUM(news_count) as news_count "
            "FROM sentiment_daily WHERE trade_date>=%s GROUP BY trade_date ORDER BY trade_date", (trend_start,))

        today_news = []
        try:
            today_news = Database.fetchall(
                "SELECT code, title, date, source, sentiment, sent_score "
                "FROM raw_news WHERE date >= %s ORDER BY date DESC LIMIT 50",
                (trend_start,))
        except Exception:
            pass

        return {"today": td[0] if td else {}, "trend": trend, "news": today_news or []}
    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}


@router.get("/sectors/rank")
async def get_sector_sentiment(
    sector_name=Query(None, min_length=1, max_length=100),
    sector_type=Query("concept"),
    snapshot_id=Query(None, min_length=32, max_length=32),
):
    if sector_type not in ("industry", "concept"):
        raise HTTPException(422, "sector_type 必须为 industry 或 concept")
    if sector_type == "industry" and (not sector_name or not snapshot_id):
        raise HTTPException(422, "行业情绪必须绑定行业快照和板块名称")
    if sector_type == "concept":
        from backend.service.concept_data import require_fresh_concept_snapshot
        try:
            await anyio.to_thread.run_sync(require_fresh_concept_snapshot)
        except ValueError as e:
            raise HTTPException(409, str(e))

    from core.data import Database
    if not Database.is_available():
        return {"data": [], "meta": {"status": "unavailable", "missing_reason": "database_unavailable"}}
    if sector_type == "industry":
        from core.data.repos.stock_industry_history import (
            get_industry_history_snapshot,
            get_industry_snapshot_member_count,
        )
        from core.shared.trading_time import market_date
        from core.shared.data_contract import make_data_envelope
        snapshot = await anyio.to_thread.run_sync(get_industry_history_snapshot, snapshot_id)
        if (not snapshot or snapshot.get("status") != "complete" or
                not snapshot.get("available_after_date") or
                not snapshot.get("source_update_date") or not snapshot.get("query_date") or
                not snapshot.get("classification_standard") or
                int(snapshot.get("row_count") or 0) < 2500 or int(snapshot.get("unique_code_count") or 0) < 2500 or
                str(snapshot["available_after_date"])[:10] > market_date().isoformat()):
            raise HTTPException(409, "行业快照缺失或尚未可用")
        as_of_date = market_date()
        if any(snapshot.get(field) and str(snapshot[field])[:10] > as_of_date.isoformat()
               for field in ("query_date", "source_update_date")):
            raise HTTPException(409, "行业快照日期晚于当前市场日期")
        start_date = as_of_date - timedelta(days=7)
        rows = await anyio.to_thread.run_sync(
            Database.fetchall,
            "SELECT r.industry AS code, AVG(sd.avg_score) AS score, "
            "SUM(sd.news_count) AS news_count, COUNT(DISTINCT sd.code) AS stock_count, "
            "MAX(sd.trade_date) AS latest_trade_date "
            "FROM sentiment_daily sd JOIN stock_industry_history_rows r ON sd.code=r.symbol "
            "WHERE r.payload_sha256=%s AND r.industry=%s "
            "AND sd.trade_date BETWEEN %s AND %s "
            "GROUP BY r.industry",
            (snapshot["payload_sha256"], sector_name, start_date, as_of_date),
        )
        result = rows or []
        member_count = await anyio.to_thread.run_sync(
            get_industry_snapshot_member_count, snapshot["payload_sha256"], sector_name,
        )
        stock_count = int(result[0].get("stock_count") or 0) if result else 0
        latest_trade_date = result[0].get("latest_trade_date") if result else None
        latest_date = (latest_trade_date.isoformat() if hasattr(latest_trade_date, "isoformat")
                       else str(latest_trade_date)[:10] if latest_trade_date else None)
        membership_age = (as_of_date - date.fromisoformat(str(snapshot["available_after_date"])[:10])).days
        membership_status = "stale" if membership_age > 14 else "available"
        stale = membership_status == "stale" or bool(latest_date and latest_date < (as_of_date - timedelta(days=7)).isoformat())
        envelope = make_data_envelope(
            source="sentiment_daily+stock_industry_history",
            observed_at=snapshot.get("observed_at"),
            available_at=latest_date,
            as_of=as_of_date.isoformat(),
            version=snapshot_id,
            coverage={"requested": member_count, "accepted": stock_count},
            status="stale" if stale else "available" if result else "unavailable",
            missing_reason="sector_sentiment_stale" if stale else None if result else "no_matching_sector_sentiment",
        )
        return {
            "data": result,
            "meta": {
                "status": envelope["status"],
                "sector_type": "industry",
                "sector_name": sector_name,
                "snapshot_id": snapshot_id,
                "source": snapshot["source_name"],
                "membership_status": membership_status,
                "latest_trade_date": latest_date,
                "coverage": {"requested": member_count, "accepted": stock_count},
                "data_envelope": envelope,
                "missing_reason": envelope.get("missing_reason"),
            },
        }
    if sector_name is None:
        rows = await anyio.to_thread.run_sync(
            Database.fetchall,
            "SELECT s.sector_name as code, AVG(sd.avg_score) as score, "
            "SUM(sd.news_count) as news_count, COUNT(DISTINCT sd.code) as stock_count "
            "FROM sentiment_daily sd JOIN stock_sectors s ON sd.code=s.code "
            "AND s.sector_type='concept' "
            "WHERE sd.trade_date>=DATE_SUB(CURDATE(), INTERVAL 5 DAY) "
            "GROUP BY s.sector_name ORDER BY score DESC LIMIT 20")
        return {"data": rows or [], "meta": {"status": "available" if rows else "unavailable",
                                             "sector_type": "concept",
                                             "missing_reason": None if rows else "no_sector_sentiment"}}
    rows = await anyio.to_thread.run_sync(
        Database.fetchall,
        "SELECT s.sector_name AS code, AVG(sd.avg_score) AS score, "
        "SUM(sd.news_count) AS news_count, COUNT(DISTINCT sd.code) AS stock_count, "
        "MAX(sd.trade_date) AS latest_trade_date "
        "FROM sentiment_daily sd JOIN stock_sectors s ON sd.code=s.code "
        "WHERE s.sector_type=%s AND s.sector_name=%s "
        "AND sd.trade_date>=DATE_SUB(CURDATE(), INTERVAL 5 DAY) "
        "GROUP BY s.sector_name",
        (sector_type, sector_name),
    )
    result = rows or []
    return {
        "data": result,
        "meta": {
            "status": "available" if result else "unavailable",
            "sector_type": sector_type,
            "sector_name": sector_name,
            "missing_reason": None if result else "no_matching_sector_sentiment",
        },
    }


