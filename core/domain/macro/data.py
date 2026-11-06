from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from core.shared.persistence import today_date
from typing import Dict, List, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MacroSnapshot:
    as_of_date: str
    shibor_on: Optional[float] = None
    shibor_1y: Optional[float] = None
    shibor_3m: Optional[float] = None
    spread_1y_on: Optional[float] = None
    lpr_1y: Optional[float] = None
    lpr_5y: Optional[float] = None
    m1_yoy: Optional[float] = None
    m2_yoy: Optional[float] = None
    m1_m2_scissors: Optional[float] = None
    cpi_yoy: Optional[float] = None
    core_cpi_yoy: Optional[float] = None
    ppi_yoy: Optional[float] = None
    ppi_mp_yoy: Optional[float] = None
    cpi_ppi_spread: Optional[float] = None
    pmi: Optional[float] = None
    pmi_new_order: Optional[float] = None
    pmi_non_mfg: Optional[float] = None
    raw: Dict = field(default_factory=dict)


class MacroDataAccess:

    @staticmethod
    def _query(table, columns, where=None, params=None, order_by=None, limit=None):
        try:
            from core.data import Database
            sql = f"SELECT {columns} FROM {table}"
            if where:
                sql += f" WHERE {where}"
            if order_by:
                sql += f" ORDER BY {order_by}"
            if limit:
                sql += f" LIMIT {limit}"
            return Database.fetchall(sql, params) or []
        except Exception as e:
            logger.warning(f"[MacroData] {table} 查询失败: {e}")
            return []

    @classmethod
    def get_latest_shibor(cls, as_of=None):
        if as_of is None:
            as_of = today_date()
        rows = cls._query("macro_shibor", "*", "trade_date <= %s", (as_of,), "trade_date DESC", 1)
        return rows[0] if rows else None

    @classmethod
    def get_latest_lpr(cls, as_of=None):
        if as_of is None:
            as_of = today_date()
        rows = cls._query("macro_lpr", "*", "trade_date <= %s", (as_of,), "trade_date DESC", 1)
        return rows[0] if rows else None

    @classmethod
    def get_latest_money_supply(cls, as_of=None):
        if as_of is None:
            as_of = today_date()
        rows = cls._query("macro_money_supply", "*", "trade_date <= %s", (as_of,), "trade_date DESC", 1)
        return rows[0] if rows else None

    @classmethod
    def get_latest_cpi(cls, as_of=None):
        if as_of is None:
            as_of = today_date()
        rows = cls._query("macro_cpi", "*", "trade_date <= %s", (as_of,), "trade_date DESC", 1)
        return rows[0] if rows else None

    @classmethod
    def get_latest_ppi(cls, as_of=None):
        if as_of is None:
            as_of = today_date()
        rows = cls._query("macro_ppi", "*", "trade_date <= %s", (as_of,), "trade_date DESC", 1)
        return rows[0] if rows else None

    @classmethod
    def get_latest_pmi(cls, as_of=None):
        if as_of is None:
            as_of = today_date()
        rows = cls._query("macro_pmi", "*", "trade_date <= %s", (as_of,), "trade_date DESC", 1)
        return rows[0] if rows else None

    @classmethod
    def _latest_published_monthly(cls, table, as_of):
        if isinstance(as_of, datetime):
            cutoff = as_of
        else:
            value = str(as_of or "").strip()
            if not value or "T" not in value and " " not in value:
                return None
            try:
                cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        rows = cls._query(table, "*")
        visible = []
        for row in rows:
            published_at = row.get("published_at") or row.get("available_at")
            if not published_at:
                continue
            try:
                published_at = (published_at if isinstance(published_at, datetime)
                                else datetime.fromisoformat(str(published_at).replace("Z", "+00:00")))
            except (TypeError, ValueError):
                continue
            if (published_at.tzinfo is None) != (cutoff.tzinfo is None):
                continue
            if published_at <= cutoff:
                visible.append((published_at, row))
        return max(visible, key=lambda item: item[0])[1] if visible else None

    @classmethod
    def get_macro_snapshot(cls, as_of=None) -> Optional[MacroSnapshot]:
        live_snapshot = as_of is None
        if as_of is None:
            as_of = today_date()
        as_of_str = as_of.isoformat() if hasattr(as_of, 'isoformat') else str(as_of)

        shibor = cls.get_latest_shibor(as_of)
        lpr = cls.get_latest_lpr(as_of)
        if live_snapshot:
            money = cls.get_latest_money_supply(as_of)
            cpi = cls.get_latest_cpi(as_of)
            ppi = cls.get_latest_ppi(as_of)
            pmi = cls.get_latest_pmi(as_of)
        else:
            money = cls._latest_published_monthly("macro_money_supply", as_of)
            cpi = cls._latest_published_monthly("macro_cpi", as_of)
            ppi = cls._latest_published_monthly("macro_ppi", as_of)
            pmi = cls._latest_published_monthly("macro_pmi", as_of)

        if not any([shibor, lpr, money, cpi, ppi, pmi]):
            return None

        snap = MacroSnapshot(as_of_date=as_of_str)

        if shibor:
            snap.shibor_on = float(shibor["on_rate"]) if shibor.get("on_rate") is not None else None
            snap.shibor_1y = float(shibor["y1_rate"]) if shibor.get("y1_rate") is not None else None
            snap.shibor_3m = float(shibor["m3_rate"]) if shibor.get("m3_rate") is not None else None
            if snap.shibor_1y is not None and snap.shibor_on is not None:
                snap.spread_1y_on = round(snap.shibor_1y - snap.shibor_on, 4)
            snap.raw["shibor"] = shibor

        if lpr:
            snap.lpr_1y = float(lpr["lpr_1y"]) if lpr.get("lpr_1y") is not None else None
            snap.lpr_5y = float(lpr["lpr_5y"]) if lpr.get("lpr_5y") is not None else None
            snap.raw["lpr"] = lpr

        if money:
            snap.m1_yoy = float(money["m1_yoy"]) if money.get("m1_yoy") is not None else None
            snap.m2_yoy = float(money["m2_yoy"]) if money.get("m2_yoy") is not None else None
            if snap.m1_yoy is not None and snap.m2_yoy is not None:
                snap.m1_m2_scissors = round(snap.m1_yoy - snap.m2_yoy, 4)
            snap.raw["money"] = money

        if cpi:
            snap.cpi_yoy = float(cpi["cpi_yoy"]) if cpi.get("cpi_yoy") is not None else None
            snap.core_cpi_yoy = float(cpi["core_cpi_yoy"]) if cpi.get("core_cpi_yoy") is not None else None
            snap.raw["cpi"] = cpi

        if ppi:
            snap.ppi_yoy = float(ppi["ppi_yoy"]) if ppi.get("ppi_yoy") is not None else None
            snap.ppi_mp_yoy = float(ppi["ppi_mp_yoy"]) if ppi.get("ppi_mp_yoy") is not None else None
            snap.raw["ppi"] = ppi

        if snap.cpi_yoy is not None and snap.ppi_yoy is not None:
            snap.cpi_ppi_spread = round(snap.cpi_yoy - snap.ppi_yoy, 4)

        if pmi:
            snap.pmi = float(pmi["pmi"]) if pmi.get("pmi") is not None else None
            snap.pmi_new_order = float(pmi["pmi_new_order"]) if pmi.get("pmi_new_order") is not None else None
            snap.pmi_non_mfg = float(pmi["pmi_non_mfg"]) if pmi.get("pmi_non_mfg") is not None else None
            snap.raw["pmi"] = pmi

        return snap

    @classmethod
    def get_macro_timeseries(cls, start, end=None) -> pd.DataFrame:
        if end is None:
            end = today_date()
        start_s = start.isoformat() if hasattr(start, 'isoformat') else str(start)
        end_s = end.isoformat() if hasattr(end, 'isoformat') else str(end)

        dfs = []

        for table, date_col, fields in [
            ("macro_shibor", "trade_date", ["on_rate", "y1_rate", "m3_rate"]),
            ("macro_lpr", "trade_date", ["lpr_1y", "lpr_5y"]),
        ]:
            cols = ", ".join([date_col] + fields)
            rows = cls._query(table, cols, f"{date_col} BETWEEN %s AND %s", (start_s, end_s), date_col)
            if rows:
                df = pd.DataFrame(rows)
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how="outer")

        # 前值填充月度数据到日频
        date_range = pd.date_range(start=start_s, end=end_s, freq="D")
        result = result.reindex(date_range, method="ffill")

        # 计算衍生指标
        if "on_rate" in result.columns and "y1_rate" in result.columns:
            result["spread_1y_on"] = result["y1_rate"] - result["on_rate"]
        if "m1_yoy" in result.columns and "m2_yoy" in result.columns:
            result["m1_m2_scissors"] = result["m1_yoy"] - result["m2_yoy"]
        if "cpi_yoy" in result.columns and "ppi_yoy" in result.columns:
            result["cpi_ppi_spread"] = result["cpi_yoy"] - result["ppi_yoy"]

        return result

    @classmethod
    def get_macro_change(cls, indicator, days=30, as_of=None) -> Optional[float]:
        end = as_of or today_date()
        if isinstance(end, str):
            end = date.fromisoformat(end[:10])
        start = end - timedelta(days=days)
        ts = cls.get_macro_timeseries(start, end)
        if ts.empty or indicator not in ts.columns:
            return None
        recent = ts[indicator].dropna()
        if len(recent) < 2:
            return None
        return float(recent.iloc[-1] - recent.iloc[0])
