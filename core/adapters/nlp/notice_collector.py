"""公告独立采集器——akshare主力 + 东财直连备用 + 巨潮兜底"""
import hashlib
import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

NOTICE_TYPES = {
    "业绩预告": ["业绩预告", "净利润", "盈利预", "业绩快报", "业绩修正"],
    "财务报告": ["年报", "半年报", "季报", "财务报告", "审计报告", "年度报告", "半年度报告", "季度报告"],
    "分红送转": ["分红", "送转", "派息", "转增", "权益分派", "利润分配", "现金分红"],
    "增减持": ["增持", "减持", "股份变动", "持股变动", "股东减持", "股东增持"],
    "重大事项": ["重组", "收购", "出售", "停牌", "复牌", "担保", "重大合同", "重大资产", "对外投资"],
    "人事变动": ["董事", "监事", "高管", "辞职", "任免", "聘任", "变更"],
}

NOTICE_TYPE_LIST = list(NOTICE_TYPES.keys()) + ["融资公告", "风险提示", "资产重组", "信息变更", "持股变动"]

_source_fail_count = {}
_source_disabled = set()
_lock = threading.Lock()


def _source_ok(tag):
    with _lock:
        if tag in _source_disabled:
            return False
        fails = _source_fail_count.get(tag, 0)
        if fails >= 20:
            _source_disabled.add(tag)
            logger.warning(f"[公告熔断] {tag} 连续失败 {fails} 次，自动禁用")
            return False
    return True


def _record_result(tag, success):
    with _lock:
        if success:
            _source_fail_count[tag] = 0
        else:
            _source_fail_count[tag] = _source_fail_count.get(tag, 0) + 1


def classify(title):
    for ntype, keywords in NOTICE_TYPES.items():
        for kw in keywords:
            if kw in title:
                return ntype
    return "其他"


class NoticeCollector:

    def __init__(self):
        self._min_interval = 0.5
        self._last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    @staticmethod
    def _normalize_date(date_str):
        """统一日期格式为 YYYY-MM-DD，兼容 YYYYMMDD / YYYY-MM-DD / YYYYMMDDHHMMSS"""
        s = str(date_str).strip().replace("-", "").replace("/", "").replace(" ", "")[:8]
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return "0000-00-00"

    def _make_item(self, code, title, url, date_str, source_tag, confidence=0.95):
        notice_type = classify(title)
        normalized_date = self._normalize_date(date_str)
        return {
            "code": code,
            "title": title.strip(),
            "url": url or "",
            "notice_date": normalized_date,
            "notice_type": notice_type,
            "source": source_tag,
            "source_confidence": confidence,
            "hash": hashlib.md5((title.strip() + normalized_date + code).encode()).hexdigest(),
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ═══════════════════ 第一层：akshare 主力源 ═══════════════════

    def _fetch_akshare(self, code, begin_date=None, end_date=None):
        if not begin_date:
            begin_date = "20250701"
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        items = []
        for ntype in NOTICE_TYPE_LIST:
            if not _source_ok("akshare_notice"):
                break
            try:
                self._rate_limit()
                from core.data.sources.akshare import call_ak
                df = call_ak("stock_individual_notice_report",
                             security=code, symbol=ntype,
                             begin_date=begin_date, end_date=end_date)
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    title = str(row.get("公告标题", row.get("title", "")))
                    if not title or title == "nan":
                        continue
                    url = str(row.get("网址", row.get("url", "")))
                    notice_date = str(row.get("公告日期", row.get("notice_date", "")))
                    if not notice_date or notice_date == "nan":
                        continue
                    items.append(self._make_item(code, title, url, notice_date, "akshare_notice"))
                _record_result("akshare_notice", True)
            except Exception as e:
                _record_result("akshare_notice", False)
                logger.debug(f"[akshare公告]({code}/{ntype}) 失败: {e}")
        return items

    # ═══════════════════ 第二层：东财直连备用 ═══════════════════

    def _fetch_eastmoney_direct(self, code):
        if not _source_ok("eastmoney_notice"):
            return []
        try:
            from core.data.sources.scraper.news_scrapers import EastmoneyNoticeScraper
            raw = EastmoneyNoticeScraper().get_news(code, pages=2)
            items = []
            for it in raw:
                items.append(self._make_item(
                    code, it["title"], it["url"], it["date"], "eastmoney_notice", 0.90))
            _record_result("eastmoney_notice", True)
            return items
        except Exception as e:
            _record_result("eastmoney_notice", False)
            logger.debug(f"[东财公告]({code}) 失败: {e}")
            return []

    # ═══════════════════ 第三层：巨潮兜底 ═══════════════════

    def _fetch_cninfo(self, code):
        if not _source_ok("cninfo_notice"):
            return []
        try:
            from core.data.sources.scraper.news_scrapers import CninfoScraper
            raw = CninfoScraper().get_news(code)
            items = []
            for it in raw:
                items.append(self._make_item(
                    code, it["title"], it["url"], it["date"], "cninfo_notice", 0.92))
            _record_result("cninfo_notice", True)
            return items
        except Exception as e:
            _record_result("cninfo_notice", False)
            logger.debug(f"[巨潮公告]({code}) 失败: {e}")
            return []

    # ═══════════════════ 汇总采集 ═══════════════════

    def collect(self, code, begin_date=None, end_date=None):
        all_items = []

        akshare_items = self._fetch_akshare(code, begin_date, end_date)
        all_items.extend(akshare_items)

        if not akshare_items:
            em_items = self._fetch_eastmoney_direct(code)
            all_items.extend(em_items)

        if len(all_items) < 5:
            cninfo_items = self._fetch_cninfo(code)
            all_items.extend(cninfo_items)

        seen = set()
        deduped = []
        for it in all_items:
            h = it["hash"]
            if h not in seen:
                seen.add(h)
                deduped.append(it)

        return deduped

    def collect_full_market(self, date_str=None):
        """按日期批量采集全市场公告，用于定时调度"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        all_items = []
        for ntype in NOTICE_TYPE_LIST:
            try:
                self._rate_limit()
                from core.data.sources.akshare import call_ak
                df = call_ak("stock_notice_report", symbol=ntype, date=date_str)
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    code = str(row.get("代码", row.get("code", ""))).strip()
                    title = str(row.get("公告标题", row.get("title", "")))
                    if not code or not title or title == "nan" or len(code) != 6:
                        continue
                    url = str(row.get("网址", row.get("url", "")))
                    notice_date = str(row.get("公告日期", row.get("notice_date", "")))
                    all_items.append(self._make_item(code, title, url, notice_date if notice_date else date_str, "akshare_notice"))
            except Exception as e:
                logger.debug(f"[全市场公告]({ntype}/{date_str}) 失败: {e}")

        seen = set()
        deduped = []
        for it in all_items:
            h = it["hash"]
            if h not in seen:
                seen.add(h)
                deduped.append(it)

        return deduped

    # ═══════════════════ 数据库操作 ═══════════════════

    def save_to_db(self, items):
        if not items:
            return 0
        try:
            from core.data import execute_sql
            saved = 0
            for item in items:
                try:
                    execute_sql(
                        "INSERT IGNORE INTO stock_notices "
                        "(code,title,url,notice_date,notice_type,source,source_confidence,hash,collected_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (item["code"], item["title"], item.get("url", ""),
                         item["notice_date"], item.get("notice_type", "其他"),
                         item.get("source", ""), item.get("source_confidence", 0.95),
                         item["hash"], item["collected_at"]))
                    saved += 1
                except Exception:
                    pass
            return saved
        except Exception as e:
            logger.warning(f"notice save_to_db: {e}")
            return 0
