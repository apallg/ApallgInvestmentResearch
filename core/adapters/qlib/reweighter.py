"""
训练样本加权 — 按流动性分档加权 + 次新股样本剔除 (权重 0)。

数据层 (bridge/tushare_client) 已剔除 ST/退市/ETF/B股, 此处补训练管道内缺失的两层:
  1. 次新: 上市 (库内首条日线) 后 60 个交易日内的样本权重 0
  2. 流动性: 训练段日均成交额 <1000万 → 0.3, 1000万~5000万 → 0.7, ≥5000万 → 1.0
接入 qlib 官方 Reweighter 入口: model.fit(dataset, reweighter=LiquidityReweighter(...))。
权重表一次 SQL 从 daily_quotes 聚合 (per-code 静态), 不逐股循环。
"""
import logging
import numpy as np
import pandas as pd
from qlib.data.dataset.weight import Reweighter

logger = logging.getLogger(__name__)

NEW_STOCK_DAYS = 60
LOW_AMOUNT = 1e7
MID_AMOUNT = 5e7


class LiquidityReweighter(Reweighter):

    def __init__(self, start_date, end_date):
        self.amount_w = {}
        self.first_date = {}
        try:
            from core.data import Database
            rows = Database.fetchall(
                "SELECT code, AVG(amount) AS avg_amt, MIN(trade_date) AS first_dt "
                "FROM daily_quotes WHERE trade_date BETWEEN %s AND %s GROUP BY code",
                (start_date, end_date),
            )
            for r in rows or []:
                code = str(r["code"])
                avg_amt = float(r["avg_amt"] or 0)
                if avg_amt < LOW_AMOUNT:
                    self.amount_w[code] = 0.3
                elif avg_amt < MID_AMOUNT:
                    self.amount_w[code] = 0.7
                else:
                    self.amount_w[code] = 1.0
                self.first_date[code] = pd.Timestamp(r["first_dt"])
            logger.info(f"[Reweighter] 权重表加载: {len(self.amount_w)} 只 "
                        f"({start_date}~{end_date})")
        except Exception as e:
            logger.warning(f"[Reweighter] 权重表加载失败, 全部等权: {e}")

    def reweight(self, data):
        idx = data.index
        n = len(idx)
        if not self.amount_w:
            return pd.Series(np.ones(n), index=idx)
        codes = [str(c)[2:] if str(c)[:2].lower() in ("sh", "sz") else str(c)
                 for c in idx.get_level_values("instrument")]
        dates = pd.DatetimeIndex(idx.get_level_values("datetime"))
        w = np.array([self.amount_w.get(c, 1.0) for c in codes])
        cutoff = np.array([
            (self.first_date.get(c) or pd.Timestamp("1990-01-01")) + pd.Timedelta(days=NEW_STOCK_DAYS)
            for c in codes
        ], dtype="datetime64[ns]")
        w = np.where(dates.values < cutoff, 0.0, w)
        n_zero = int((w == 0).sum())
        if n_zero:
            logger.info(f"[Reweighter] 次新样本置零: {n_zero}/{n} 行")
        return pd.Series(w, index=idx)
