"""板块 regime 检测 — 流通市值加权合成行业指数 → 每行业市场状态 → 与大盘背离表

行业指数从 daily_quotes 合成（成分股日收益按最新流通市值加权 → 累积成指数），
不依赖外部行业指数数据源。复用 MarketRegimeClassifier.classify_from_df 逐行业分类。
"""
import datetime
import logging
from core.shared.persistence import today_date

import pandas as pd

from core.shared import config
from core.domain.regime.classifier import MarketRegimeClassifier, RegimeResult
from core.domain.regime.weights import get_regime_direction

logger = logging.getLogger(__name__)

SECTOR_REGIME_KEY = "sector_regime:latest"
SECTOR_REGIME_TTL = 1800  # 实时缓存 30 分钟
MIN_SECTOR_STOCKS = 8     # 成分股过少的行业不判定
DIVERGENCE_MIN_CONF = float(getattr(config, "SECTOR_DIVERGENCE_MIN_CONF", 0.5))


def prefetch_sector_quotes(start, end):
    """预拉取全市场 daily_quotes 日线 (历史确定数据, 回测引擎一次性拉取复用)。

    返回 cast 后的 polars DataFrame (Float64 价格 + Date trade_date),
    供 _build_sector_index_frames 逐调仓日切片, 避免 N 次全市场大查询。
    """
    import polars as pl
    from core.data import Database
    if not Database.is_available():
        return None
    start_s = start.isoformat() if hasattr(start, 'isoformat') else str(start)
    end_s = end.isoformat() if hasattr(end, 'isoformat') else str(end)
    rows = Database.fetchall(
        "SELECT code, trade_date, open, high, low, close, volume FROM daily_quotes "
        "WHERE trade_date >= %s AND trade_date <= %s ORDER BY code, trade_date",
        (start_s, end_s))
    if not rows:
        return None
    df = pl.from_dicts(rows, infer_schema_length=len(rows) + 1)
    return df.with_columns([pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")] +
                           [pl.col("trade_date").cast(pl.Date)])


def _build_sector_index_frames(as_of=None, lookback_days=None, quotes_df=None):
    """合成各行业流通市值加权指数, 返回 {industry: df(trade_date/open/high/low/close/volume)}

    quotes_df: 预拉取的全市场 daily_quotes polars DataFrame (回测引擎复用)。
               None 时走现场拉取。
    """
    from core.data import Database
    from core.data.fetcher import DataFetcher

    if not Database.is_available():
        return {}
    lookback_days = lookback_days or getattr(config, "REGIME_LOOKBACK_DAYS", 252)

    ind_map_raw = DataFetcher().get_industry_map() or {}
    code_industry = {c: v.get("industry") for c, v in ind_map_raw.items()
                     if isinstance(v, dict) and v.get("industry")}
    if not code_industry:
        logger.warning("[SectorRegime] 行业映射为空")
        return {}

    end = as_of or today_date()
    if isinstance(end, str):
        end = datetime.date.fromisoformat(end[:10])
    start = end - datetime.timedelta(days=int(lookback_days * 1.6))

    # 最新流通市值做静态权重（行业内部权重结构变化缓慢, 静态足够）
    mv_rows = Database.fetchall(
        "SELECT code, circ_mv FROM daily_basic "
        "WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= %s) "
        "AND circ_mv > 0",
        (end,))
    mv = {r["code"]: float(r["circ_mv"]) for r in mv_rows}
    if not mv:
        logger.warning("[SectorRegime] daily_basic 无流通市值数据")
        return {}

    import polars as pl
    if quotes_df is not None:
        # 复用预拉取的全市场日线, 切片到 [start, end] (仅取 <= end, 无前视)
        df = quotes_df.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        if df.is_empty():
            return {}
    else:
        rows = Database.fetchall(
            "SELECT code, trade_date, open, high, low, close, volume FROM daily_quotes "
            "WHERE trade_date >= %s AND trade_date <= %s ORDER BY code, trade_date",
            (start, end))
        if not rows:
            return {}
        df = pl.from_dicts(rows, infer_schema_length=len(rows) + 1)
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")])
    ind_df = pl.DataFrame({"code": list(code_industry.keys()),
                           "industry": list(code_industry.values())})
    w_df = pl.DataFrame({"code": list(mv.keys()), "w": list(mv.values())})
    df = df.join(ind_df, on="code", how="inner").join(w_df, on="code", how="inner")
    df = df.filter(pl.col("close") > 0)

    # pct_chg 入库常为 NULL, 用 close 自算日收益; 剔除 |r|>21% 脏数据(复权跳变等)
    df = df.sort(["code", "trade_date"]).with_columns(
        pl.col("close").shift(1).over("code").alias("prev_close"))
    df = df.filter(pl.col("prev_close") > 0)
    df = df.with_columns([
        (pl.col("close") / pl.col("prev_close") - 1).alias("r_c"),
        (pl.col("open") / pl.col("prev_close") - 1).alias("r_o"),
        (pl.col("high") / pl.col("prev_close") - 1).alias("r_h"),
        (pl.col("low") / pl.col("prev_close") - 1).alias("r_l"),
    ])
    df = df.filter(pl.col("r_c").abs() < 0.21)

    agg = (df.group_by(["industry", "trade_date"])
           .agg([
               ((pl.col("r_c") * pl.col("w")).sum() / pl.col("w").sum()).alias("rc"),
               ((pl.col("r_o") * pl.col("w")).sum() / pl.col("w").sum()).alias("ro"),
               ((pl.col("r_h") * pl.col("w")).sum() / pl.col("w").sum()).alias("rh"),
               ((pl.col("r_l") * pl.col("w")).sum() / pl.col("w").sum()).alias("rl"),
               pl.col("volume").sum().alias("volume"),
               pl.len().alias("n"),
           ])
           .sort(["industry", "trade_date"]))

    frames = {}
    for name, g in agg.group_by(["industry"], maintain_order=True):
        industry = name[0] if isinstance(name, tuple) else name
        gp = g.to_pandas()
        if len(gp) < 60 or gp["n"].median() < MIN_SECTOR_STOCKS:
            continue
        close = 1000.0 * (1 + gp["rc"]).cumprod()
        prev = close.shift(1).fillna(1000.0)
        frames[industry] = pd.DataFrame({
            "trade_date": gp["trade_date"],
            "open": prev * (1 + gp["ro"]),
            "high": prev * (1 + gp["rh"]),
            "low": prev * (1 + gp["rl"]),
            "close": close,
            "volume": gp["volume"],
        })
    logger.info(f"[SectorRegime] 行业指数合成: {len(frames)} 个行业 "
                f"({start} ~ {end}, 市值权重 {len(mv)} 只)")
    return frames


def get_sector_regimes(as_of=None, quotes_df=None):
    """逐行业市场状态 {industry: RegimeResult}。
    as_of 给定(历史场景)时强制实时计算, 不读写缓存, 防未来函数。
    quotes_df: 预拉取的全市场 daily_quotes (回测复用)。
    """
    from core.shared.infrastructure import redis_client as rds

    if as_of is None:
        cached = rds._get(SECTOR_REGIME_KEY)
        if cached and isinstance(cached, dict):
            try:
                return {k: RegimeResult.from_dict(v) for k, v in cached.items()}
            except Exception:
                pass

    frames = _build_sector_index_frames(as_of=as_of, quotes_df=quotes_df)
    results = {}
    for industry, df in frames.items():
        try:
            results[industry] = MarketRegimeClassifier(index_name=f"行业:{industry}").classify_from_df(df)
        except Exception as e:
            logger.debug(f"[SectorRegime] {industry} 分类失败: {e}")

    if as_of is None and results:
        try:
            rds._set(SECTOR_REGIME_KEY,
                     {k: v.to_dict() for k, v in results.items()},
                     expire=SECTOR_REGIME_TTL)
        except Exception:
            pass
    return results


def get_divergent_sectors(as_of=None, market_regime=None, quotes_df=None):
    """与大盘策略方向背离的行业 {industry: {"direction","regime","confidence"}}

    背离 = 行业方向(REGIME_DIRECTION 映射) ≠ 市场方向 且 行业置信度 ≥ 阈值。
    任何异常返回 {} (调用方自动降级单模型)。
    quotes_df: 预拉取的全市场 daily_quotes (回测复用)。
    """
    try:
        if market_regime is None:
            from core.domain.regime.unified import get_unified_regime
            market_regime = get_unified_regime(as_of=as_of, apply_macro=True)
        market_dir = get_regime_direction(market_regime.regime)

        out = {}
        for industry, rr in get_sector_regimes(as_of=as_of, quotes_df=quotes_df).items():
            if rr.confidence < DIVERGENCE_MIN_CONF:
                continue
            d = get_regime_direction(rr.regime)
            if d != market_dir:
                out[industry] = {"direction": d, "regime": rr.regime.value,
                                 "confidence": round(float(rr.confidence), 3)}
        if out:
            brief = {k: f"{v['regime']}/{v['confidence']}" for k, v in list(out.items())[:8]}
            logger.info(f"[SectorRegime] 背离板块 {len(out)} 个 "
                        f"(市场={market_regime.regime.value}→{market_dir}): {brief}")
        return out
    except Exception as e:
        logger.warning(f"[SectorRegime] 背离检测失败, 降级单模型: {e}")
        return {}
