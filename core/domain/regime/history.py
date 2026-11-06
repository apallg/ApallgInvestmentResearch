import logging
from datetime import date, timedelta
from core.shared.persistence import today_date

logger = logging.getLogger(__name__)


def backfill(start: date = None, end: date = None):
    """批量回填历史市场状态到 market_regime_history 表。

    使用确定性规则分类器逐交易日生成6状态，已存在日期按同口径覆盖。
    """
    import pandas as pd
    from core.data import Database

    if not Database.is_available():
        logger.warning("[RegimeHistory] 数据库不可用, 跳过回填")
        return {"backfilled": 0, "message": "数据库不可用"}

    from core.data.fetcher import DataFetcher
    df = DataFetcher().get_index_daily("沪深300", days=2500)
    if df is None or df.empty:
        logger.warning("[RegimeHistory] 无法获取指数数据")
        return {"backfilled": 0, "message": "无法获取指数数据"}

    if "trade_date" not in df.columns and "date" in df.columns:
        df["trade_date"] = df["date"]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    from core.data.repos.regime_history import regime_history_repo
    from core.domain.regime.volatility_history import build_volatility_history
    regime_history_repo.ensure_schema()
    rows = build_volatility_history(df, start=start, end=end)
    written = 0
    for index in range(0, len(rows), 500):
        written += regime_history_repo.upsert(rows[index:index + 500])
    logger.info(f"[RegimeHistory] 逐交易日回填完成: {written} 条")
    return {"backfilled": written}


def ensure_today():
    """确保今日市场状态与可审计波动轴已入库。"""
    from core.data import Database

    if not Database.is_available():
        return

    try:
        from core.data.fetcher import DataFetcher
        from core.data.repos.regime_history import regime_history_repo
        from core.domain.regime.volatility_history import build_volatility_history
        from core.domain.regime.classifier import get_current_regime
        rr = get_current_regime()
        today_str = today_date().isoformat()
        frame = DataFetcher().get_index_daily("沪深300", days=1200)
        rows = build_volatility_history(frame, start=today_str, end=today_str)
        if not rows:
            logger.warning(f"[RegimeHistory] {today_str} 无可用交易日波动状态")
            return
        row = rows[-1]
        row["regime"] = rr.regime.value
        row["confidence"] = float(rr.confidence)
        row["method"] = "unified+rule_vol_pit_v1"
        regime_history_repo.ensure_schema()
        regime_history_repo.upsert([row])
    except Exception as exc:
        logger.warning(f"[RegimeHistory] 今日状态更新失败: {exc}")


def get_db_regime():
    """快速查当前最新市场状态 (从 DB 读，不回退实时计算)。"""
    from core.data import Database
    if not Database.is_available():
        return None
    try:
        rows = Database.fetchall(
            "SELECT regime, confidence FROM market_regime_history "
            "ORDER BY trade_date DESC LIMIT 1"
        )
        if rows:
            return {"regime": rows[0]["regime"], "confidence": float(rows[0]["confidence"])}
    except Exception:
        pass
    return None


def get_regime_map(date_list):
    """给定日期列表，返回 {date_str: regime} 映射 (从 DB 批量读取)。"""
    from core.data import Database
    if not Database.is_available() or not date_list:
        return {}
    try:
        date_strs = [d.isoformat() if hasattr(d, "isoformat") else str(d)[:10] for d in date_list]
        placeholders = ",".join(["%s"] * len(date_strs))
        rows = Database.fetchall(
            f"SELECT trade_date, regime FROM market_regime_history "
            f"WHERE trade_date IN ({placeholders})",
            tuple(date_strs),
        )
        return {str(r["trade_date"]): r["regime"] for r in rows}
    except Exception:
        return {}
