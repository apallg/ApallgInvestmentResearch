"""
板块选股器 - 按行业板块筛选股票，按成交量排序
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import akshare as ak
from core.domain.screening.density_analyzer import find_dense_zones, classify_zones, calc_risk_reward
from core.shared.infrastructure.cache import CacheManager
from core.shared import config

logger = logging.getLogger(__name__)

_SIGNAL_THRESHOLD = 3       # 支撑/压力距离阈值（%）
_STRONG_SIGNAL_THRESHOLD = 2  # 强信号阈值（%）


class SectorScanner:
    """板块扫描器"""

    def __init__(self, cache: Optional[CacheManager] = None):
        self.cache = cache or CacheManager()
        self._sector_map = None
        self._sector_map_loaded = False
        self._sector_map_industry = None
        self._sector_map_concept = None

    def _load_sector_map(self, sector_type='industry') -> dict:
        """从 MySQL stock_sectors 表加载板块映射。sector_type: 'industry' 或 'concept'"""
        if sector_type == 'industry' and self._sector_map_loaded:
            return self._sector_map or {}
        if sector_type == 'concept' and self._sector_map_concept is not None:
            return self._sector_map_concept or {}

        try:
            from core.data import Database
            if Database.is_available():
                rows = Database.fetchall(
                    "SELECT code, sector_name FROM stock_sectors WHERE sector_type=%s ORDER BY sector_name",
                    (sector_type,))
                if rows:
                    sector_map = {}
                    for r in rows:
                        sn = r["sector_name"]
                        if sn not in sector_map:
                            sector_map[sn] = {"codes": [], "source": "mysql", "count": 0}
                        sector_map[sn]["codes"].append(r["code"])
                    for sn in sector_map:
                        sector_map[sn]["count"] = len(sector_map[sn]["codes"])
                    if sector_type == 'industry':
                        self._sector_map = sector_map
                        self._sector_map_loaded = True
                    else:
                        self._sector_map_concept = sector_map
                    logger.info(f"已从 MySQL 加载板块映射: {len(sector_map)} 个 ({sector_type})")
                    return sector_map
        except Exception as e:
            logger.warning(f"加载板块映射失败 ({sector_type}): {e}")

        empty = {}
        if sector_type == 'industry':
            self._sector_map = empty
            self._sector_map_loaded = True
        else:
            self._sector_map_concept = empty
        return empty

    def get_all_sectors(self) -> pd.DataFrame:
        """获取所有行业板块列表"""
        from core.data.fetcher import _retry
        cache_key = "sectors_list_v2"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        sector_ttl = getattr(config, 'CACHE_EXPIRE_SECTOR_LIST', 900)
        # 优先同花顺 (不走东财域名，不受代理影响)
        try:
            df = _retry(lambda: ak.stock_board_industry_name_ths(), ma=2, dl=3)
            if df is not None and not df.empty:
                col = "name" if "name" in df.columns else df.columns[0]
                df = df.rename(columns={col: "板块名称"})
                self.cache.set(cache_key, df, expire=sector_ttl)
                return df
        except Exception as e:
            logger.warning(f"THS 板块列表失败: {e}")
        # 备选：东财
        try:
            df = _retry(lambda: ak.stock_board_industry_name_em(), ma=3, dl=3)
            self.cache.set(cache_key, df, expire=sector_ttl)
            return df
        except Exception as e:
            logger.warning(f"东财板块列表失败: {e}")
        # 最后备选：硬编码常见板块
        common = ["电力行业", "半导体", "白酒", "医药生物", "汽车整车",
                  "银行", "证券", "房地产", "国防军工", "计算机应用",
                  "通信设备", "新能源汽车", "光伏设备", "医疗器械",
                  "食品饮料", "煤炭开采", "有色金属", "建筑装饰",
                  "石油石化", "机械设备", "基础化工", "传媒",
                  "电子", "家电", "纺织服装", "商贸零售",
                  "交通运输", "农林牧渔", "环保", "综合"]
        logger.info(f"使用硬编码板块列表后备 ({len(common)}个板块)")
        fallback = pd.DataFrame({"板块名称": common})
        self.cache.set(cache_key, fallback, expire=120)
        return fallback

    def get_sector_stocks(self, sector_name: str, sector_type='industry') -> pd.DataFrame:
        """获取某个板块的所有成分股，按成交量降序排列。sector_type: 'industry' 或 'concept'"""
        cache_key = f"sector_stocks_v7:{sector_type}:{sector_name}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        sector_ttl = getattr(config, 'CACHE_EXPIRE_SECTOR_STOCKS', 300)

        sector_map = self._load_sector_map(sector_type)
        info = sector_map.get(sector_name) if sector_map else None
        if not info or not info.get("codes"):
            return pd.DataFrame()

        df = pd.DataFrame({
            "代码": info["codes"],
            "名称": [""] * len(info["codes"]),
            "最新价": [np.nan] * len(info["codes"]),
            "涨跌幅": [np.nan] * len(info["codes"]),
            "成交量": [np.nan] * len(info["codes"]),
        })
        logger.info(f"本地板块映射: {sector_name} → {len(df)}只 (来源: {info.get('source')})")

        # 统一列名
        col_map = {
            "代码": "code", "名称": "name", "最新价": "price",
            "涨跌幅": "pct_chg", "涨跌额": "change", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover",
            "市盈率-动态": "pe", "市净率": "pb",
            "振幅": "amplitude", "最高": "high", "最低": "low",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 按成交量降序排序
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df = df.sort_values("volume", ascending=False).reset_index(drop=True)

        # 添加排名
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        df = df.reset_index()

        self.cache.set(cache_key, df, expire=sector_ttl)
        return df

    def get_concept_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """获取概念板块成分股，从同花顺详情页抓取（带缓存）"""
        cache_key = f"concept_sector_stocks_v1:{sector_name}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        sector_ttl = getattr(config, 'CACHE_EXPIRE_SECTOR_STOCKS', 300)

        try:
            # 获取概念板块代码
            concept_df = ak.stock_board_concept_name_ths()
            name_col = "name" if "name" in concept_df.columns else concept_df.columns[0]
            code_col = "code" if "code" in concept_df.columns else concept_df.columns[1]
            match = concept_df[concept_df[name_col] == sector_name]
            if match.empty:
                return pd.DataFrame()
            concept_code = str(match.iloc[0][code_col])

            # 获取THS认证cookie (复用 akshare_utils 单例防 V8 多线程冲突)
            from core.data.sources.akshare import get_mini_racer
            from akshare.datasets import get_ths_js
            from core.shared.persistence import load_text
            js_code = get_mini_racer()
            js_content = load_text(get_ths_js("ths.js"))
            js_code.eval(js_content)
            v_code = js_code.call("v")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Cookie": f"v={v_code}",
            }

            # 爬取首页获取总页数
            url = f"http://q.10jqka.com.cn/gn/detail/code/{concept_code}/"
            r = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(r.text, "lxml")
            page_span = soup.find(name="span", attrs={"class": "page_info"})
            total_pages = int(page_span.text.strip().split("/")[1]) if page_span else 1

            all_codes = []
            for page in range(1, total_pages + 1):
                if page == 1:
                    page_soup = soup
                else:
                    page_url = f"http://q.10jqka.com.cn/gn/detail/code/{concept_code}/order/desc/page/{page}/ajax/1/"
                    r = requests.get(page_url, headers=headers, timeout=15)
                    page_soup = BeautifulSoup(r.text, "lxml")
                table = page_soup.find("table")
                if not table:
                    continue
                rows = table.find_all("tr")[1:]
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) >= 2:
                        code_text = cols[1].get_text(strip=True)
                        if code_text.isdigit() and len(code_text) == 6:
                            all_codes.append(code_text)

            if not all_codes:
                return pd.DataFrame()

            df = pd.DataFrame({
                "code": all_codes,
                "name": [""] * len(all_codes),
                "price": [0.0] * len(all_codes),
                "pct_chg": [0.0] * len(all_codes),
                "volume": [0] * len(all_codes),
            })
            df = df.sort_values("volume", ascending=False).reset_index(drop=True)
            logger.info(f"概念板块: {sector_name} → {len(df)}只 (来源: 10jqka)")

            self.cache.set(cache_key, df, expire=sector_ttl)
            return df
        except Exception as e:
            logger.warning(f"获取概念板块成分股失败: {sector_name} → {e}")
            return pd.DataFrame()

    def scan_support_resistance(self, sector_name: str, max_stocks: int = 80, member_rows=None) -> list:
        """批量扫描板块成分股的支撑位/压力位，返回接近关键位置的股票"""
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        from core.data.fetcher import DataFetcher
        from core.domain.screening.indicators import Analyzer

        stocks_df = (self.get_sector_stocks(sector_name) if member_rows is None else
                     pd.DataFrame([{"code": row.get("code"), "name": row.get("name") or ""}
                                   for row in member_rows if row.get("code")]))
        if stocks_df is None or stocks_df.empty:
            return []

        code_col = "code" if "code" in stocks_df.columns else "代码"
        name_col = "name" if "name" in stocks_df.columns else "名称"
        codes = stocks_df[code_col].head(max_stocks).tolist()

        def _analyze_one(code):
            try:
                fetcher = DataFetcher()
                df = fetcher.get_stock_daily(code)
                if df is None or df.empty or len(df) < 20:
                    return None
                df = Analyzer.add_indicators(df)
                latest = df.iloc[-1]
                recent = df.tail(20)

                price = float(latest["close"])
                if price <= 0:
                    return None

                bb_lower = float(latest.get("BB_lower", 0))
                bb_upper = float(latest.get("BB_upper", 0))
                ma20 = float(latest.get("MA20", 0))
                ma60 = float(latest.get("MA60", 0))
                low20 = float(recent["low"].min())
                high20 = float(recent["high"].max())

                # 支撑位列表 (只取低于当前价的)
                supports = []
                if bb_lower > 0 and bb_lower < price:
                    supports.append(("布林下轨", bb_lower))
                if ma60 > 0 and ma60 < price:
                    supports.append(("MA60均线", ma60))
                if low20 < price:
                    supports.append(("20日最低", low20))
                if ma20 > 0 and ma20 < price:
                    supports.append(("MA20均线", ma20))

                # 压力位列表 (只取高于当前价的)
                resistances = []
                if bb_upper > 0 and bb_upper > price:
                    resistances.append(("布林上轨", bb_upper))
                if ma20 > 0 and ma20 > price:
                    resistances.append(("MA20均线", ma20))
                if high20 > price:
                    resistances.append(("20日最高", high20))
                if ma60 > 0 and ma60 > price:
                    resistances.append(("MA60均线", ma60))

                nearest_support = min(supports, key=lambda x: price - x[1]) if supports else None
                nearest_resistance = min(resistances, key=lambda x: x[1] - price) if resistances else None

                dist_support = round((price - nearest_support[1]) / price * 100, 2) if nearest_support else None
                dist_resistance = round((nearest_resistance[1] - price) / price * 100, 2) if nearest_resistance else None

                # 信号判定
                signal = ""
                signal_type = ""
                if dist_support is not None and dist_support <= _SIGNAL_THRESHOLD:
                    if nearest_support[0] == "布林下轨" and dist_support <= _STRONG_SIGNAL_THRESHOLD:
                        signal = "强支撑买入"
                        signal_type = "strong_buy"
                    else:
                        signal = "接近支撑"
                        signal_type = "near_support"
                elif dist_resistance is not None and dist_resistance <= _SIGNAL_THRESHOLD:
                    if nearest_resistance[0] == "布林上轨" and dist_resistance <= _STRONG_SIGNAL_THRESHOLD:
                        signal = "强压力卖出"
                        signal_type = "strong_sell"
                    else:
                        signal = "接近压力"
                        signal_type = "near_resistance"

                return {
                    "code": str(code),
                    "name": "",
                    "price": round(price, 2),
                    "support_name": nearest_support[0] if nearest_support else "",
                    "support_price": round(nearest_support[1], 2) if nearest_support else 0,
                    "dist_support": dist_support if dist_support is not None else 999,
                    "resistance_name": nearest_resistance[0] if nearest_resistance else "",
                    "resistance_price": round(nearest_resistance[1], 2) if nearest_resistance else 0,
                    "dist_resistance": dist_resistance if dist_resistance is not None else 999,
                    "signal": signal,
                    "signal_type": signal_type,
                    "pct_chg": float(latest["pct_chg"]) if pd.notna(latest.get("pct_chg")) else None,
                }
            except Exception:
                return None

        results = []
        with thread_pool.default.borrow() as pool:
            futures = {pool.submit(_analyze_one, c): c for c in codes}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    # 补股票名称
                    name_row = stocks_df[stocks_df[code_col] == r["code"]]
                    if not name_row.empty:
                        r["name"] = str(name_row.iloc[0].get(name_col, ""))
                    results.append(r)

        # 排序：有信号的优先，离支撑/压力越近越靠前
        signal_priority = {"strong_buy": 0, "near_support": 1, "strong_sell": 2, "near_resistance": 3, "": 4}
        results.sort(key=lambda x: (
            signal_priority.get(x["signal_type"], 4),
            min(x["dist_support"], x["dist_resistance"])
        ))
        return results

    def scan_dense_zones(
        self, sector_name: str, max_stocks: int = 80,
        window: int = 40, n_zones: int = 3, member_rows=None,
    ) -> dict:
        """
        板块密集成交区扫描

        对板块成分股逐只计算成交密集区（支撑/压力位），
        按盈亏比排序返回，同时汇总板块级别统计。

        返回:
            {sector, total, scanned, support_pct, resistance_pct,
             neutral_pct, avg_rr, status, stocks: [...]}
        """
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        from core.data.fetcher import DataFetcher

        stocks_df = (self.get_sector_stocks(sector_name) if member_rows is None else
                     pd.DataFrame([{"code": row.get("code"), "name": row.get("name") or ""}
                                   for row in member_rows if row.get("code")]))
        if stocks_df is None or stocks_df.empty:
            return {"sector": sector_name, "total": 0, "scanned": 0, "stocks": []}

        code_col = "code" if "code" in stocks_df.columns else "代码"
        name_col = "name" if "name" in stocks_df.columns else "名称"
        codes = stocks_df[code_col].head(max_stocks).tolist()

        threshold = getattr(config, 'DENSE_ZONE_PRICE_THRESHOLD_PCT', 2.0)

        def _analyze_one(code):
            try:
                fetcher = DataFetcher()
                df = fetcher.get_stock_daily(code)
                if df is None or df.empty or len(df) < window + 5:
                    return None

                current_price = float(df["close"].iloc[-1])
                zones = find_dense_zones(df, window=window, n_zones=n_zones)
                classify_zones(zones, current_price)
                rr = calc_risk_reward(current_price, zones)

                supports = [z for z in zones if z.zone_type == "support"]
                resistances = [z for z in zones if z.zone_type == "resistance"]

                nearest_support = min(supports, key=lambda z: abs(z.distance_pct)) if supports else None
                nearest_resistance = min(resistances, key=lambda z: abs(z.distance_pct)) if resistances else None

                support_dist = abs(nearest_support.distance_pct) if nearest_support else None
                resistance_dist = abs(nearest_resistance.distance_pct) if nearest_resistance else None

                near_support = support_dist is not None and support_dist <= threshold
                near_resistance = resistance_dist is not None and resistance_dist <= threshold

                top_zones = []
                for z in zones[:3]:
                    top_zones.append({
                        "center": z.center, "low": z.low, "high": z.high,
                        "strength": z.strength, "type": z.zone_type,
                        "dist_pct": z.distance_pct, "touch_count": z.touch_count,
                        "vol_pct": z.volume_pct,
                    })

                return {
                    "code": str(code),
                    "price": round(current_price, 2),
                    "nearest_support": nearest_support.center if nearest_support else None,
                    "nearest_resistance": nearest_resistance.center if nearest_resistance else None,
                    "support_dist_pct": round(support_dist, 2) if support_dist else None,
                    "resistance_dist_pct": round(resistance_dist, 2) if resistance_dist else None,
                    "rr_ratio": rr["risk_reward_ratio"],
                    "rr_quality": rr["quality"],
                    "zone_count": len(zones),
                    "near_support": near_support,
                    "near_resistance": near_resistance,
                    "top_zones": top_zones,
                }
            except Exception:
                return None

        stock_results = []
        near_support_count = 0
        near_resistance_count = 0
        rr_values = []

        with thread_pool.default.borrow() as pool:
            futures = {pool.submit(_analyze_one, c): c for c in codes}
            for fut in as_completed(futures):
                r = fut.result()
                if r is None:
                    continue
                name_row = stocks_df[stocks_df[code_col] == r["code"]]
                if not name_row.empty:
                    r["name"] = str(name_row.iloc[0].get(name_col, ""))
                else:
                    r["name"] = r["code"]

                if r["near_support"]:
                    near_support_count += 1
                if r["near_resistance"]:
                    near_resistance_count += 1
                rr_values.append(r["rr_ratio"])
                stock_results.append(r)

        scanned = len(stock_results)
        total = len(codes)

        support_pct = round(near_support_count / scanned * 100, 1) if scanned else 0
        resistance_pct = round(near_resistance_count / scanned * 100, 1) if scanned else 0
        neutral_pct = round(100 - support_pct - resistance_pct, 1)
        avg_rr = round(np.mean(rr_values), 2) if rr_values else 0

        # 板块状态
        if support_pct > resistance_pct + 10:
            status = "偏多"
        elif resistance_pct > support_pct + 10:
            status = "偏空"
        else:
            status = "均衡"

        # 按盈亏比降序
        stock_results.sort(key=lambda x: x["rr_ratio"], reverse=True)

        return {
            "sector": sector_name,
            "total": total,
            "scanned": scanned,
            "support_pct": support_pct,
            "resistance_pct": resistance_pct,
            "neutral_pct": neutral_pct,
            "avg_rr": avg_rr,
            "status": status,
            "stocks": stock_results,
        }

    def _collect_sector_stocks(self, sector_names, max_per_sector=30):
        """收集全板块候选股票，去重并记录所属板块和名称。返回 (stock_sectors, stock_names)。"""
        stock_sectors: dict = {}
        stock_names: dict = {}
        for sname in sector_names:
            try:
                sdf = self.get_sector_stocks(sname)
                if sdf is None or sdf.empty:
                    continue
                code_col = "code" if "code" in sdf.columns else "代码"
                name_col = "name" if "name" in sdf.columns else "名称"
                for row in sdf.head(max_per_sector).itertuples(index=False):
                    c = str(getattr(row, code_col, ''))
                    if c not in stock_sectors:
                        stock_sectors[c] = []
                        stock_names[c] = str(getattr(row, name_col, ''))
                    stock_sectors[c].append(sname)
            except Exception as e:
                logger.debug(f"_collect_sector_stocks({sname}): {e}")
        return stock_sectors, stock_names

    def scan_all_sectors_sr(
        self, max_per_sector: int = 30, max_sectors: int = 30,
        max_results: int = 200, window: int = 40,
    ) -> list:
        """
        全板块支撑/压力位扫描

        遍历所有板块，对每板块成交量前 max_per_sector 只股做支撑/压力分析，
        只返回有交易信号的股票，跨板块混合并按信号优先级排序。

        返回:
            [{code, name, sector, price, pct_chg, support_name, support_price,
              dist_support, resistance_name, resistance_price, dist_resistance,
              signal, signal_type}, ...]
        """
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        from core.data.fetcher import DataFetcher
        from core.domain.screening.indicators import Analyzer

        sectors_df = self.get_all_sectors()
        if sectors_df is None or sectors_df.empty:
            return []

        sector_col = sectors_df.columns[0]
        sector_names = sectors_df[sector_col].head(max_sectors).tolist()

        stock_sectors, _stock_names = self._collect_sector_stocks(sector_names, max_per_sector)
        if not stock_sectors:
            return []

        threshold = _SIGNAL_THRESHOLD
        strong_threshold = _STRONG_SIGNAL_THRESHOLD

        def _analyze_one(code):
            try:
                fetcher = DataFetcher()
                df = fetcher.get_stock_daily(code)
                if df is None or df.empty or len(df) < 20:
                    return None
                df = Analyzer.add_indicators(df)
                latest = df.iloc[-1]
                recent = df.tail(20)

                price = float(latest["close"])
                if price <= 0:
                    return None

                bb_lower = float(latest.get("BB_lower", 0))
                bb_upper = float(latest.get("BB_upper", 0))
                ma20 = float(latest.get("MA20", 0))
                ma60 = float(latest.get("MA60", 0))
                low20 = float(recent["low"].min())
                high20 = float(recent["high"].max())

                supports = []
                if bb_lower > 0 and bb_lower < price:
                    supports.append(("布林下轨", bb_lower))
                if ma60 > 0 and ma60 < price:
                    supports.append(("MA60均线", ma60))
                if low20 < price:
                    supports.append(("20日最低", low20))
                if ma20 > 0 and ma20 < price:
                    supports.append(("MA20均线", ma20))

                resistances = []
                if bb_upper > 0 and bb_upper > price:
                    resistances.append(("布林上轨", bb_upper))
                if ma20 > 0 and ma20 > price:
                    resistances.append(("MA20均线", ma20))
                if high20 > price:
                    resistances.append(("20日最高", high20))
                if ma60 > 0 and ma60 > price:
                    resistances.append(("MA60均线", ma60))

                nearest_support = min(supports, key=lambda x: price - x[1]) if supports else None
                nearest_resistance = min(resistances, key=lambda x: x[1] - price) if resistances else None

                dist_support = round((price - nearest_support[1]) / price * 100, 2) if nearest_support else None
                dist_resistance = round((nearest_resistance[1] - price) / price * 100, 2) if nearest_resistance else None

                signal = ""
                signal_type = ""
                if dist_support is not None and dist_support <= threshold:
                    if nearest_support[0] == "布林下轨" and dist_support <= strong_threshold:
                        signal = "强支撑买入"
                        signal_type = "strong_buy"
                    else:
                        signal = "接近支撑"
                        signal_type = "near_support"
                elif dist_resistance is not None and dist_resistance <= threshold:
                    if nearest_resistance[0] == "布林上轨" and dist_resistance <= strong_threshold:
                        signal = "强压力卖出"
                        signal_type = "strong_sell"
                    else:
                        signal = "接近压力"
                        signal_type = "near_resistance"

                if not signal:
                    return None

                return {
                    "code": code,
                    "sectors": stock_sectors.get(code, []),
                    "price": round(price, 2),
                    "pct_chg": float(latest.get("pct_chg", 0)),
                    "support_name": nearest_support[0] if nearest_support else "",
                    "support_price": round(nearest_support[1], 2) if nearest_support else 0,
                    "dist_support": dist_support if dist_support is not None else 999,
                    "resistance_name": nearest_resistance[0] if nearest_resistance else "",
                    "resistance_price": round(nearest_resistance[1], 2) if nearest_resistance else 0,
                    "dist_resistance": dist_resistance if dist_resistance is not None else 999,
                    "signal": signal,
                    "signal_type": signal_type,
                }
            except Exception:
                return None

        results = []
        with thread_pool.default.borrow() as pool:
            futures = {pool.submit(_analyze_one, c): c for c in stock_sectors}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)

        signal_priority = {"strong_buy": 0, "near_support": 1, "strong_sell": 2, "near_resistance": 3}
        results.sort(key=lambda x: (
            signal_priority.get(x["signal_type"], 4),
            min(x["dist_support"], x["dist_resistance"])
        ))
        return results[:max_results]

    def scan_all_sectors_dz(
        self, max_per_sector: int = 30, max_sectors: int = 30,
        max_results: int = 200, window: int = 40, n_zones: int = 3,
    ) -> list:
        """
        全板块密集成交区扫描

        遍历所有板块，对每板块成交量前 max_per_sector 只股做密集区分析，
        按盈亏比降序排列，跨板块混合。

        返回:
            [{code, name, sectors, price, nearest_support, nearest_resistance,
              support_dist_pct, resistance_dist_pct, rr_ratio, rr_quality,
              zone_count, top_zones}, ...]
        """
        from concurrent.futures import as_completed
        from core.shared.infrastructure.thread_pool import thread_pool
        from core.data.fetcher import DataFetcher

        sectors_df = self.get_all_sectors()
        if sectors_df is None or sectors_df.empty:
            return []

        sector_col = sectors_df.columns[0]
        sector_names = sectors_df[sector_col].head(max_sectors).tolist()

        stock_sectors, stock_names = self._collect_sector_stocks(sector_names, max_per_sector)
        if not stock_sectors:
            return []

        threshold = getattr(config, 'DENSE_ZONE_PRICE_THRESHOLD_PCT', 2.0)

        def _analyze_one(code):
            try:
                fetcher = DataFetcher()
                df = fetcher.get_stock_daily(code)
                if df is None or df.empty or len(df) < window + 5:
                    return None

                current_price = float(df["close"].iloc[-1])
                zones = find_dense_zones(df, window=window, n_zones=n_zones)
                classify_zones(zones, current_price)
                rr = calc_risk_reward(current_price, zones)

                supports = [z for z in zones if z.zone_type == "support"]
                resistances = [z for z in zones if z.zone_type == "resistance"]

                nearest_support = min(supports, key=lambda z: abs(z.distance_pct)) if supports else None
                nearest_resistance = min(resistances, key=lambda z: abs(z.distance_pct)) if resistances else None

                support_dist = abs(nearest_support.distance_pct) if nearest_support else None
                resistance_dist = abs(nearest_resistance.distance_pct) if nearest_resistance else None

                top_zones = []
                for z in zones[:3]:
                    top_zones.append({
                        "center": z.center, "low": z.low, "high": z.high,
                        "strength": z.strength, "type": z.zone_type,
                        "dist_pct": z.distance_pct, "touch_count": z.touch_count,
                    })

                return {
                    "code": code,
                    "name": stock_names.get(code, code),
                    "sectors": stock_sectors.get(code, []),
                    "price": round(current_price, 2),
                    "nearest_support": nearest_support.center if nearest_support else None,
                    "nearest_resistance": nearest_resistance.center if nearest_resistance else None,
                    "support_dist_pct": round(support_dist, 2) if support_dist else None,
                    "resistance_dist_pct": round(resistance_dist, 2) if resistance_dist else None,
                    "rr_ratio": rr["risk_reward_ratio"],
                    "rr_quality": rr["quality"],
                    "zone_count": len(zones),
                    "top_zones": top_zones,
                }
            except Exception:
                return None

        results = []
        with thread_pool.default.borrow() as pool:
            futures = {pool.submit(_analyze_one, c): c for c in stock_sectors}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)

        results.sort(key=lambda x: x["rr_ratio"], reverse=True)
        return results[:max_results]

    @staticmethod
    def format_sector_context(sector_name: str, df: pd.DataFrame, top_n: int = 10) -> str:
        """生成供 LLM 分析的板块上下文"""
        top = df.head(top_n)
        lines = [
            f"板块名称: {sector_name}",
            f"成分股总数: {len(df)}",
            "",
            f"=== 成交量前 {top_n} 的个股 ===",
            "",
        ]
        for row in top.itertuples(index=False):
            lines.append(
                f"{getattr(row, 'rank', '?')}. {getattr(row, 'name', '')}({getattr(row, 'code', '')}) "
                f"最新价:{getattr(row, 'price', 'N/A')} "
                f"涨跌:{getattr(row, 'pct_chg', 'N/A')}% "
                f"成交量:{getattr(row, 'volume', 'N/A')} "
                f"换手率:{getattr(row, 'turnover', 'N/A')}%"
            )

        return "\n".join(lines)

    @staticmethod
    def format_sector_summary(sector_name: str, df: pd.DataFrame) -> str:
        """生成板块统计摘要"""
        total = len(df)
        up_count = len(df[df.get("pct_chg", 0).astype(float) > 0]) if "pct_chg" in df.columns else 0
        down_count = total - up_count
        avg_pct = df["pct_chg"].astype(float).mean() if "pct_chg" in df.columns else 0
        total_vol = df["volume"].sum() if "volume" in df.columns else 0

        return (
            f"【{sector_name}】{total}只成分股，"
            f"上涨{up_count}只，下跌{down_count}只，"
            f"平均涨跌幅{avg_pct:.2f}%，"
            f"板块总成交量{total_vol:.0f}万手"
        )
