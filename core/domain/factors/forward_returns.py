"""
统一 forward return 计算和缓存层。
替代 overfitting_checker._batch_forward_returns 和 config_tuner._compute_forward_returns
两份重复代码。

三层缓存:
  L0: 进程内存 dict {(code, trade_date, days): float}
  L1: MySQL 物化表 daily_forward_ret (预计算)
  L2: 实时计算 (作为 warm_up 未覆盖日期的回退)
"""

import logging
import datetime as _dt
import threading

logger = logging.getLogger(__name__)

__all__ = ["ForwardReturnCache", "get_forward_cache"]


class ForwardReturnCache:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._mem = {}  # (code, date_iso, days) → float (L0)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def warm_up(self, start_date, end_date, periods=(5, 10, 20, 40)):
        """批量预计算所有股票的 forward return，写入 daily_forward_ret 物化表。
        使用 MySQL 8.0 LEAD 窗口函数一次完成全市场计算，替代逐股 Python 循环。
        """
        from core.data import Database
        if not Database.is_available():
            logger.warning("MySQL 不可用, 跳过 forward return 预热")
            return False

        try:
            Database.execute("""
                CREATE TABLE IF NOT EXISTS daily_forward_ret (
                    code VARCHAR(10) NOT NULL,
                    trade_date DATE NOT NULL,
                    fwd_5d DECIMAL(10,6) DEFAULT NULL,
                    fwd_10d DECIMAL(10,6) DEFAULT NULL,
                    fwd_20d DECIMAL(10,6) DEFAULT NULL,
                    fwd_40d DECIMAL(10,6) DEFAULT NULL,
                    PRIMARY KEY (code, trade_date),
                    KEY idx_date (trade_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        except Exception:
            pass

        period_cols = {5: "fwd_5d", 10: "fwd_10d", 20: "fwd_20d", 40: "fwd_40d"}
        # 只对请求的 periods 构建 LEAD 表达式
        lead_parts = []
        for p in sorted(periods):
            col = period_cols.get(p)
            if col:
                # entry = 次日开盘价, exit = p 日后收盘价
                # LEAD(open, 1) = 下一交易日开盘, LEAD(close, p+1) = p+1个交易日后收盘
                lead_parts.append(
                    f"COALESCE(NULLIF(LEAD(open,{p+1}) OVER w, 0), "
                    f"LEAD(close,{p}) OVER w) / NULLIF(LEAD(open,1) OVER w, 0) - 1 "
                    f"AS {col}"
                )

        if not lead_parts:
            return False

        expr = ", ".join(lead_parts)
        sql = (
            f"INSERT INTO daily_forward_ret (code, trade_date, {', '.join(period_cols[p] for p in sorted(periods) if p in period_cols)}) "
            f"SELECT code, trade_date, {expr} "
            f"FROM daily_quotes "
            f"WHERE trade_date >= %s AND trade_date <= %s "
            f"WINDOW w AS (PARTITION BY code ORDER BY trade_date) "
            f"ON DUPLICATE KEY UPDATE "
            + ", ".join(f"{period_cols[p]}=VALUES({period_cols[p]})"
                        for p in sorted(periods) if p in period_cols)
        )

        try:
            start = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
            end = end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date)
            Database.execute(sql, [start, end])
            logger.info(f"forward return 预热完成: {start} ~ {end}")
            return True
        except Exception as e:
            logger.warning(f"forward return 预热失败 (可能 MySQL < 8.0 无 LEAD 支持): {e}")
            return False

    def get_batch(self, codes, eval_date, forward_days):
        """批量获取 forward return。优先 L0 内存 → L1 物化表。
        返回 {code: float}，缺失的 code 不包含在结果中。
        """
        if len(codes) == 0:
            return {}
        date_key = eval_date.isoformat() if hasattr(eval_date, "isoformat") else str(eval_date)

        result = {}
        missing = []
        for c in codes:
            key = (c, date_key, forward_days)
            v = self._mem.get(key)
            if v is not None:
                result[c] = v
            else:
                missing.append(c)

        if not missing:
            return result

        col_map = {5: "fwd_5d", 10: "fwd_10d", 20: "fwd_20d", 40: "fwd_40d"}
        col = col_map.get(forward_days)
        if col:
            try:
                from core.data import Database
                if Database.is_available():
                    ph = ",".join(["%s"] * len(missing))
                    rows = Database.fetchall(
                        f"SELECT code, {col} AS fwd FROM daily_forward_ret "
                        f"WHERE code IN ({ph}) AND trade_date = %s",
                        missing + [date_key],
                    )
                    for r in rows:
                        v = float(r["fwd"]) if r["fwd"] is not None else 0.0
                        key = (r["code"], date_key, forward_days)
                        self._mem[key] = v
                        result[r["code"]] = v
            except Exception as e:
                logger.debug(f"[FwdRet] L1 缓存查询失败 ({len(missing)} codes): {e}")

        # L2 实时计算: DB 值为 NULL 的, 用 daily_quotes LEAD 窗口实时算
        need_calc = [c for c in missing if c not in result]
        if need_calc:
            try:
                from core.data import Database
                if Database.is_available():
                    ph = ",".join(["%s"] * len(need_calc))
                    # 与 warm_up (物化表) 同一口径: t+1 开盘进场 → t+p+1 开盘出场 (open 缺失回退收盘)
                    # 修复 L2 close-to-close 与物化表 open-to-open 不一致导致的 IC 标签跨日漂移
                    sql = (
                        f"SELECT code, entry_px, exit_px FROM ("
                        f"  SELECT code, trade_date, "
                        f"    LEAD(open, 1) OVER w AS entry_px, "
                        f"    COALESCE(NULLIF(LEAD(open, {forward_days}+1) OVER w, 0), "
                        f"            LEAD(close, {forward_days}) OVER w) AS exit_px "
                        f"  FROM daily_quotes "
                        f"  WHERE code IN ({ph}) "
                        f"  WINDOW w AS (PARTITION BY code ORDER BY trade_date)"
                        f") t WHERE trade_date = %s AND entry_px > 0 AND exit_px > 0"
                    )
                    rows = Database.fetchall(sql, need_calc + [date_key])
                    for r in rows:
                        c = r["code"]
                        entry_px = float(r["entry_px"]) if r["entry_px"] else 0
                        exit_px = float(r["exit_px"]) if r["exit_px"] else 0
                        if entry_px > 0 and exit_px > 0:
                            fwd = exit_px / entry_px - 1.0
                            key = (c, date_key, forward_days)
                            self._mem[key] = fwd
                            result[c] = fwd
            except Exception as e:
                logger.debug(f"[FwdRet] L2 实时计算失败 ({len(need_calc)} codes): {e}")

        for c in missing:
            if c not in result:
                result[c] = 0.0

        return result

    def get_list(self, codes, eval_date, forward_days):
        """兼容旧 _batch_forward_returns 接口: 返回与 codes 顺序一致的列表。"""
        d = self.get_batch(codes, eval_date, forward_days)
        return [d.get(c, 0.0) for c in codes]

    def get_multi_period(self, codes, eval_date, periods=(5, 10, 20)):
        """兼容旧 _compute_forward_returns 接口: 返回 {period: {code: float}}。"""
        result = {}
        for p in periods:
            result[p] = self.get_batch(codes, eval_date, p)
        return result

    def clear_memory(self):
        self._mem.clear()


def get_forward_cache():
    return ForwardReturnCache.get_instance()
