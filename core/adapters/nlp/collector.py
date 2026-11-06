"""情绪数据采集器——4源：东财/新浪/公告/财联社（网易/证券时报/巨潮已失效移除）"""
import hashlib
import re
import time
import threading
from concurrent.futures import wait
from datetime import datetime
import logging
from core.shared.infrastructure.thread_pool import thread_pool

logger = logging.getLogger(__name__)

# ── 来源定义 (tag, 中文名, 类型) ──
SOURCES = [
    ("eastmoney", "东方财富", "akshare"),
    ("sina",      "新浪财经", "scraper"),
    ("notice",    "公司公告", "scraper"),
    ("cailian",   "财联社",   "akshare"),
]

SOURCE_CONFIDENCE = {
    "notice":  0.95,
    "sina":    0.55,
    "eastmoney": 0.50,
    "cailian":  0.60,
}


def record_source_health(source_tag, source_name, status, error_msg=None,
                         stock_count=0, news_count=0, elapsed_ms=0):
    """记录数据源采集状态到数据库"""
    with _source_lock:
        if status == "error":
            _source_fail_count[source_tag] = _source_fail_count.get(source_tag, 0) + 1
        else:
            _source_fail_count[source_tag] = 0
    try:
        from core.data import execute_sql
        execute_sql(
            "INSERT INTO source_health (source_tag, source_name, status, error_msg, "
            "stock_count, news_count, elapsed_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (source_tag, source_name, status, str(error_msg)[:500] if error_msg else None,
             stock_count, news_count, int(elapsed_ms)),
        )
    except Exception:
        pass


# ── 源熔断：连续失败 50 次则自动禁用 ──
_source_fail_count = {}
_source_disabled = set()
_source_lock = threading.Lock()
CIRCUIT_KEY = "circuit:state"


def _init_circuit_from_redis():
    global _source_fail_count, _source_disabled
    from core.shared.infrastructure import redis_client as rds
    if not rds.redis_enabled():
        return
    try:
        state = rds._get(CIRCUIT_KEY)
        if state and isinstance(state, dict):
            _source_fail_count = state.get("fail_count", {})
            _source_disabled = set(state.get("disabled", []))
            if _source_disabled:
                logger.info(f"[熔断] 从 Redis 恢复: 禁用={_source_disabled}")
    except Exception:
        pass


def _save_circuit_to_redis():
    from core.shared.infrastructure import redis_client as rds
    if not rds.redis_enabled():
        return
    rds._set(CIRCUIT_KEY, {
        "fail_count": dict(_source_fail_count),
        "disabled": list(_source_disabled),
    }, expire=600)


_circuit_redis_inited = False


def _init_circuit_from_redis_if_needed():
    global _circuit_redis_inited
    if _circuit_redis_inited:
        return
    _circuit_redis_inited = True
    _init_circuit_from_redis()


def _source_ok(tag):
    _init_circuit_from_redis_if_needed()
    with _source_lock:
        if tag in _source_disabled:
            return False
        fails = _source_fail_count.get(tag, 0)
        if fails >= 50:
            _source_disabled.add(tag)
            _save_circuit_to_redis()
            logger.warning(f"[熔断] {tag} 连续失败 {fails} 次，自动禁用")
            return False
    return True


# ── 公司名→代码映射（惰性加载）──
_name_to_code = None
_name_to_code_lock = threading.Lock()


NAME_MAP_KEY = "stock:name_to_code"


def _load_name_map():
    global _name_to_code
    if _name_to_code is not None:
        return _name_to_code
    with _name_to_code_lock:
        if _name_to_code is not None:
            return _name_to_code
        from core.shared.infrastructure import redis_client as rds
        cached = rds._get(NAME_MAP_KEY)
        if cached and isinstance(cached, dict):
            _name_to_code = cached
            return _name_to_code
        _name_to_code = {}
        try:
            from core.data.sources.akshare import call_ak
            df = call_ak("stock_info_a_code_name")
            for _, row in df.iterrows():
                code = str(row.get("code", "")).strip()
                name = str(row.get("name", "")).strip()
                if len(code) == 6 and code.isdigit() and name:
                    _name_to_code[name] = code
                    if len(name) >= 3:
                        short = name[-2:] if len(name) == 3 else name[2:]
                        if short not in _name_to_code:
                            _name_to_code[short] = code
        except Exception as e:
            logger.warning(f"公司名映射加载失败: {e}")
        rds._set(NAME_MAP_KEY, _name_to_code, expire=86400)
        return _name_to_code


def _find_codes_in_title(title):
    """从标题中提取所有相关股票代码（代码匹配+公司名匹配）。

    返回 set，一条新闻可能对应多只股票。
    """
    codes = set()
    title = str(title)

    # 1. 正则直接匹配6位股票代码
    for m in re.finditer(r'\b(6[0-9]{5}|0[0-9]{5}|3[0-9]{5}|2[0-9]{5}|5[0-9]{5})\b', title):
        codes.add(m.group(1))

    # 2. 公司简称匹配
    name_map = _load_name_map()
    if name_map:
        # 按名称长度降序匹配，避免 "中国平安" 被 "平安" 误匹配
        for name in sorted(name_map.keys(), key=len, reverse=True):
            if name in title and len(name) >= 2:
                codes.add(name_map[name])
    return codes


# ── 财联社全市场缓存 ──
_cailian_cache = None
_cailian_cache_time = None
_cailian_lock = threading.Lock()


# ── 线程级 Scraper 复用（复用 HTTP Session，减少 TCP 握手）──
_scraper_local = threading.local()


def _get_scrapers():
    if not hasattr(_scraper_local, 'sina'):
        from core.data.sources.scraper.sina_news import SinaNewsScraper
        _scraper_local.sina = SinaNewsScraper(min_interval=1.0)
    return _scraper_local.sina


class SentimentCollector:
    def __init__(self, cailian_cache=None):
        self._min_interval = 0.3
        self._last_request = 0
        self._cailian_snapshot = cailian_cache

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    # ═══════════════════ 东财 ═══════════════════
    # akshare stock_news_em 硬编码 pageSize=10，每只股一次拿 10 条跨 5-7 天
    # 补偿策略：同时用代码+名称两次搜索，合并去重，翻倍产出

    def _fetch_eastmoney(self, code):
        from core.data.sources.akshare import call_ak

        all_items = []
        seen_titles = set()

        def _extract(df):
            if df is None or df.empty:
                return []
            items = []
            for _, row in df.iterrows():
                title = str(row.get("新闻标题", row.get("标题", row.get("title", ""))))
                if not title or title == "nan":
                    continue
                clean_title = title.strip()
                if clean_title in seen_titles:
                    continue
                seen_titles.add(clean_title)
                pub_time = str(row.get("发布时间", row.get("pub_time", "")))
                items.append({
                    "title": clean_title,
                    "url": str(row.get("新闻链接", row.get("链接", row.get("url", "")))),
                    "date": pub_time[:10] if pub_time else "",
                    "source_tag": "eastmoney",
                    "source_confidence": SOURCE_CONFIDENCE["eastmoney"],
                })
            return items

        # 1. 按代码搜索
        try:
            df = call_ak("stock_news_em", symbol=code)
            all_items.extend(_extract(df))
        except Exception as e:
            logger.debug(f"东财代码搜索({code}): {e}")

        # 2. 按股票名称搜索（名称搜索往往返回更多公司专属新闻）
        name = self._code_to_name(code)
        if name and name != code:
            try:
                df2 = call_ak("stock_news_em", symbol=name)
                all_items.extend(_extract(df2))
            except Exception as e:
                logger.debug(f"东财名称搜索({name}): {e}")

        return all_items

    _code_to_name_cache = None

    @classmethod
    def _get_code_to_name(cls):
        if cls._code_to_name_cache is not None:
            return cls._code_to_name_cache
        name_map = _load_name_map()
        reverse = {}
        for name, c in name_map.items():
            if c not in reverse or len(name) > len(reverse[c]):
                reverse[c] = name
        cls._code_to_name_cache = reverse
        return reverse

    @classmethod
    def _code_to_name(cls, code):
        return cls._get_code_to_name().get(code)

    # ═══════════════════ 新浪 ═══════════════════

    def _fetch_sina(self, code):
        sina_scraper = _get_scrapers()
        raw = sina_scraper.get_news(code, pages=2)
        if not raw:
            return []
        items = []
        for it in raw:
            title = str(it.get("title", ""))
            if not title or title == "nan":
                continue
            items.append({
                "title": title.strip(),
                "url": str(it.get("url", "")),
                "date": str(it.get("date", it.get("time", "")))[:10],
                "source_tag": "sina",
                "source_confidence": SOURCE_CONFIDENCE["sina"],
            })
        return items

    # ═══════════════════ 公司公告 (东财HTTP直连) ═══════════════════

    def _fetch_notice(self, code):
        from core.data.sources.scraper.news_scrapers import EastmoneyNoticeScraper
        return EastmoneyNoticeScraper().get_news(code)

    # ═══════════════════ 财联社（akshare 全市场缓存）═══════════════════

    CAILIAN_KEY = "cailian:full_index"

    @staticmethod
    def _fetch_cailian_all():
        global _cailian_cache, _cailian_cache_time
        with _cailian_lock:
            now_t = time.time()
            if _cailian_cache is not None and _cailian_cache_time and (now_t - _cailian_cache_time) < 300:
                return _cailian_cache

            from core.shared.infrastructure import redis_client as rds
            cached = rds._get(SentimentCollector.CAILIAN_KEY)
            if cached is not None:
                _cailian_cache = cached
                _cailian_cache_time = now_t
                return cached
            try:
                from core.data.sources.akshare import call_ak
                df = call_ak("stock_news_main_cx")
                if df is None or df.empty:
                    return {}
                cache = {}
                for _, row in df.iterrows():
                    title = str(row.get("title", row.get("summary", row.get("name", ""))))
                    if not title or title == "nan":
                        continue
                    matched_codes = _find_codes_in_title(title)
                    if not matched_codes:
                        continue
                    item = {
                        "title": title.strip(),
                        "url": str(row.get("url", row.get("link", ""))),
                        "date": str(row.get("date", row.get("time", "")))[:10],
                        "source_tag": "cailian",
                        "source_confidence": SOURCE_CONFIDENCE["cailian"],
                    }
                    if not item["date"] or item["date"] == "nan":
                        match = re.search(r"/(\d{4}-\d{2}-\d{2})/", item["url"])
                        item["date"] = match.group(1) if match else ""
                    for c in matched_codes:
                        cache.setdefault(c, []).append(item)
                _cailian_cache = cache
                _cailian_cache_time = now_t
                rds._set(SentimentCollector.CAILIAN_KEY, cache, expire=300)
                logger.info(f"[财联社] 全市场电报索引: {sum(len(v) for v in cache.values())} 条, 覆盖 {len(cache)} 只")
                return cache
            except Exception as e:
                logger.warning(f"财联社电报: {e}")
                return _cailian_cache or {}

    def _fetch_cailian(self, code):
        cache = self._cailian_snapshot
        if cache is None:
            cache = self._fetch_cailian_all()
        return cache.get(code, [])

    def collect_global_index(self):
        cache = self._fetch_cailian_all()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        items = []
        seen = set()
        for code, code_items in cache.items():
            for source_item in code_items:
                item = dict(source_item)
                item["code"] = code
                item["collected_at"] = now
                if not item.get("date"):
                    match = re.search(r"/(\d{4}-\d{2}-\d{2})/", item.get("url", ""))
                    item["date"] = match.group(1) if match else ""
                item["hash"] = hashlib.md5(
                    (item["title"] + item.get("date", "") + "cailian").encode()
                ).hexdigest()[:12]
                key = (code, item["hash"])
                if key not in seen:
                    seen.add(key)
                    items.append(item)
        return items

    def collect_market_snapshot(self, stock_limit=20):
        items = self.collect_global_index()
        codes = []
        try:
            from core.data import Database
            from core.data.repos.local_ledger import ACCOUNT_ID
            holdings = Database.fetchall(
                "SELECT DISTINCT code FROM holdings WHERE account_id=%s AND shares > 0 LIMIT %s",
                (ACCOUNT_ID, stock_limit),
            ) or []
            codes.extend(str(row["code"]) for row in holdings)
            if len(codes) < stock_limit:
                liquid = Database.fetchall(
                    "SELECT code FROM daily_quotes WHERE trade_date=(SELECT MAX(trade_date) FROM daily_quotes) "
                    "AND code REGEXP '^(60|68|00|30|83|87|92)' "
                    "ORDER BY amount DESC LIMIT %s",
                    (stock_limit,),
                ) or []
                for row in liquid:
                    code = str(row["code"])
                    if code not in codes:
                        codes.append(code)
                    if len(codes) >= stock_limit:
                        break
        except Exception as e:
            logger.warning(f"新闻补采股票池加载失败: {e}")
        if codes:
            items.extend(self.collect(codes))
        return self.dedup(items)

    # ═══════════════════ 汇总采集（逐源独立，错误隔离）═══════════════════

    def _try_source(self, code, source_tag, source_name, fetch_fn):
        """单个源的采集包装：计时、异常捕获、健康记录"""
        t0 = time.time()
        try:
            items = fetch_fn(code)
            elapsed = (time.time() - t0) * 1000
            # 只要没抛异常就算成功（无新闻也是 ok）
            if items:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for it in items:
                    it["code"] = code
                    it["collected_at"] = now
                    if "source_tag" not in it:
                        it["source_tag"] = source_tag
                    if "source_confidence" not in it:
                        it["source_confidence"] = SOURCE_CONFIDENCE.get(source_tag, 0.50)
                    it["hash"] = hashlib.md5(
                        (it["title"] + it.get("date", "") + it["source_tag"]).encode()
                    ).hexdigest()[:12]
            record_source_health(source_tag, source_name, "ok",
                                 stock_count=1, news_count=len(items), elapsed_ms=int(elapsed))
            return items
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            record_source_health(source_tag, source_name, "error", error_msg=str(e),
                                 stock_count=1, news_count=0, elapsed_ms=int(elapsed))
            logger.debug(f"[{source_name}]({code}) 失败: {e}")
            return []

    def _fetch_all_for_stock(self, code):
        sources = []
        if _source_ok("eastmoney"):
            sources.append(("eastmoney", "东方财富", lambda c: self._fetch_eastmoney(c)))
        if _source_ok("sina"):
            sources.append(("sina", "新浪财经", lambda c: self._fetch_sina(c)))

        futures = [thread_pool.default.submit(self._try_source, code, tag, name, fn)
                   for tag, name, fn in sources]
        wait(futures)

        all_items = []
        for f in futures:
            all_items.extend(f.result())
        return all_items

    def collect_single(self, code):
        self._rate_limit()
        items = self._fetch_all_for_stock(code)
        # 追加财联社匹配
        cl_items = self._fetch_cailian(code)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for it in cl_items:
            it["code"] = code
            it["collected_at"] = now
            it["hash"] = hashlib.md5(
                (it["title"] + it.get("date", "") + "cailian").encode()
            ).hexdigest()[:12]
        return items + cl_items

    def collect(self, codes):
        # 预加载财联社全市场缓存
        try:
            self._fetch_cailian_all()
        except Exception:
            pass

        all_items = []
        for code in codes:
            items = self._fetch_all_for_stock(code)
            # 追加财联社匹配
            cl_items = self._fetch_cailian(code)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for it in cl_items:
                it["code"] = code
                it["collected_at"] = now
                it["hash"] = hashlib.md5(
                    (it["title"] + it.get("date", "") + "cailian").encode()
                ).hexdigest()[:12]
            all_items.extend(items + cl_items)
        return all_items

    def dedup(self, items):
        seen = set()
        return [i for i in items if not (i["hash"] in seen or seen.add(i["hash"]))]

    def select_for_analysis(self, items, analysis_date):
        hashes = [item.get("hash") for item in items if item.get("hash")]
        if not hashes:
            return items
        try:
            from core.data import Database
            existing = set()
            for i in range(0, len(hashes), 500):
                chunk = hashes[i:i + 500]
                placeholders = ",".join(["%s"] * len(chunk))
                rows = Database.fetchall(
                    f"SELECT hash FROM raw_news WHERE hash IN ({placeholders})",
                    tuple(chunk),
                ) or []
                existing.update(row["hash"] for row in rows)
        except Exception as e:
            logger.warning(f"新闻增量去重查询失败，回退全量分析: {e}")
            return items
        return [
            item for item in items
            if item.get("hash") not in existing
            or str(item.get("date", ""))[:10] == analysis_date
        ]

    def save_to_db(self, items):
        if not items:
            return 0
        try:
            from core.data import Database
            rows = [
                (item["code"], item["title"], item.get("url", ""), item.get("date", ""),
                 item.get("source_tag", ""), item["hash"],
                 item.get("sentiment", "neutral"), item.get("sentiment_score", 0),
                 item["collected_at"])
                for item in items
            ]
            def _do_save(cur):
                cur.executemany(
                    "INSERT IGNORE INTO raw_news (code,title,url,date,source,hash,sentiment,sent_score,collected_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    rows,
                )
            Database._execute_with_retry(lambda cur: _do_save(cur) or True, "SELECT 1", commit=True)
            return len(rows)
        except Exception as e:
            logger.warning(f"save_to_db: {e}")
            return 0

    def get_latest_news(self, code, limit=20):
        try:
            from core.data import query_sql
            return query_sql(
                "SELECT * FROM raw_news WHERE code=%s ORDER BY date DESC LIMIT %s",
                (code, limit)) or []
        except Exception as e:
            logger.warning(f"get_latest_news: {e}")
            return []

    def get_all_latest(self):
        try:
            from core.data import query_sql
            return query_sql("SELECT * FROM raw_news ORDER BY date DESC LIMIT 50") or []
        except Exception as e:
            logger.warning(f"get_all_latest: {e}")
            return []
