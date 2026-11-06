import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_stock_context_tables():
    from core.data import Database

    Database.execute("""CREATE TABLE IF NOT EXISTS stock_identity (
        ts_code VARCHAR(12) NOT NULL PRIMARY KEY, symbol VARCHAR(10), name VARCHAR(64), area VARCHAR(32),
        industry VARCHAR(64), market VARCHAR(32), exchange VARCHAR(8), list_status VARCHAR(4),
        list_date DATE, delist_date DATE, is_hs VARCHAR(2), updated_at DATE NOT NULL,
        industry_source VARCHAR(64), industry_classification VARCHAR(64), industry_as_of DATE,
        KEY idx_identity_updated (updated_at), KEY idx_identity_industry (industry)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    Database.execute("""CREATE TABLE IF NOT EXISTS stock_identity_observations (
        observed_at DATETIME(6) NOT NULL, ts_code VARCHAR(12) NOT NULL, symbol VARCHAR(10),
        name VARCHAR(64), area VARCHAR(32), industry VARCHAR(64), market VARCHAR(32),
        exchange VARCHAR(8), list_status VARCHAR(4), list_date DATE, delist_date DATE,
        is_hs VARCHAR(2), PRIMARY KEY (observed_at,ts_code), KEY idx_identity_observation_code (ts_code,observed_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    Database.execute("""CREATE TABLE IF NOT EXISTS stock_index_membership (
        ts_code VARCHAR(12) NOT NULL, index_code VARCHAR(16) NOT NULL, index_name VARCHAR(64),
        con_code VARCHAR(12), in_date DATE, out_date DATE, is_new VARCHAR(2), updated_at DATE NOT NULL,
        PRIMARY KEY (ts_code, index_code, in_date), KEY idx_membership_index (index_code),
        KEY idx_membership_updated (updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    Database.execute("""CREATE TABLE IF NOT EXISTS stock_index_membership_observations (
        observed_at DATETIME(6) NOT NULL, ts_code VARCHAR(12) NOT NULL,
        index_code VARCHAR(16) NOT NULL, index_name VARCHAR(64), in_date DATE NOT NULL,
        out_date DATE, is_new VARCHAR(2),
        PRIMARY KEY (observed_at,ts_code,index_code,in_date),
        KEY idx_observation_asof (ts_code,observed_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    Database.execute("""CREATE TABLE IF NOT EXISTS stock_dividend_events (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY, ts_code VARCHAR(12) NOT NULL, end_date DATE NOT NULL, ann_date DATE, div_proc VARCHAR(32),
        stk_div DECIMAL(18,6), cash_div DECIMAL(18,6), cash_div_tax DECIMAL(18,6), record_date DATE,
        ex_date DATE, pay_date DATE, imp_ann_date DATE, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_dividend_version (ts_code, end_date, ann_date, div_proc), KEY idx_dividend_ann (ann_date), KEY idx_dividend_ex (ex_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    from core.data.repos.dividend_source_versions import ensure_dividend_source_versions
    ensure_dividend_source_versions()
    from core.data.stock_identity_migration import ensure_stock_identity_source_fields
    ensure_stock_identity_source_fields()


def _date(value):
    if value is None or str(value) in ("", "nan", "None"):
        return None
    value = str(value).replace("-", "")[:8]
    return datetime.datetime.strptime(value, "%Y%m%d").date().isoformat()


def collect_stock_identity():
    from core.data import Database
    from core.data.sources.tushare import _rate_limit, get_pro

    ensure_stock_context_tables()
    pro = get_pro()
    if not pro:
        raise RuntimeError("TuShare 不可用")
    rows = []
    observations = []
    status_counts = {}
    observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    for status in ("L", "D", "P"):
        _rate_limit()
        df = pro.stock_basic(exchange="", list_status=status,
                             fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs")
        if df is None:
            raise RuntimeError(f"TuShare stock_basic({status}) 返回 None")
        required_columns = {"ts_code", "symbol", "name", "area", "industry", "market", "exchange",
                            "list_status", "list_date", "delist_date", "is_hs"}
        if not required_columns.issubset(df.columns):
            raise RuntimeError(f"TuShare stock_basic({status}) 缺少字段: {sorted(required_columns - set(df.columns))}")
        if df.empty and status in ("L", "D"):
            raise RuntimeError(f"TuShare stock_basic({status}) 意外返回空快照")
        status_counts[status] = len(df)
        for r in df.to_dict("records"):
            if not r.get("ts_code") or str(r.get("list_status") or "") != status:
                raise RuntimeError(f"TuShare stock_basic({status}) 存在无证券代码或状态不匹配记录")
            industry = r.get("industry")
            if industry is None or str(industry).strip().lower() in ("", "nan", "none"):
                industry = None
            values = (r.get("ts_code"), r.get("symbol"), r.get("name"), r.get("area"), industry,
                      r.get("market"), r.get("exchange"), r.get("list_status"), _date(r.get("list_date")),
                      _date(r.get("delist_date")), r.get("is_hs"))
            rows.append(values + (observed_at.date().isoformat(),))
            observations.append((observed_at,) + values)
    codes = [row[0] for row in rows]
    if status_counts.get("L", 0) < 4000 or status_counts.get("D", 0) == 0 or len(codes) != len(set(codes)):
        raise RuntimeError(f"TuShare stock_basic 快照覆盖异常: {status_counts}, unique={len(set(codes))}/{len(codes)}")
    def _save(cursor):
        cursor.executemany("""INSERT INTO stock_identity_observations
            (observed_at,ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", observations)
        cursor.executemany("""INSERT INTO stock_identity
            (ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs,updated_at,
             industry_source,industry_classification,industry_as_of)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'tushare.stock_basic','tushare.stock_basic industry',%s) ON DUPLICATE KEY UPDATE
            symbol=VALUES(symbol),name=VALUES(name),area=VALUES(area),market=VALUES(market),
            exchange=VALUES(exchange),list_status=VALUES(list_status),list_date=VALUES(list_date),
            industry=IF(VALUES(industry) IS NULL OR VALUES(industry)='',industry,VALUES(industry)),
            industry_source=IF(VALUES(industry) IS NULL OR VALUES(industry)='',industry_source,'tushare.stock_basic'),
            industry_classification=IF(VALUES(industry) IS NULL OR VALUES(industry)='',industry_classification,'tushare.stock_basic industry'),
            industry_as_of=IF(VALUES(industry) IS NULL OR VALUES(industry)='',industry_as_of,VALUES(industry_as_of)),
            delist_date=VALUES(delist_date),is_hs=VALUES(is_hs),updated_at=VALUES(updated_at)""",
            [row + (row[-1],) for row in rows])
        return len(rows)
    return Database.transaction(_save)


def collect_stock_identity_industry_fallback():
    import socket
    from core.data import Database
    from core.data.fetcher import DataFetcher

    ensure_stock_context_tables()
    targets = Database.fetchall("""SELECT ts_code,symbol FROM stock_identity
        WHERE list_status='L' AND ((industry IS NULL OR industry='')
            OR industry_source='baostock.query_stock_industry') ORDER BY ts_code""")
    if not targets:
        return {"status": "complete", "source": "baostock.query_stock_industry", "target_count": 0, "filled_count": 0}

    import baostock as bs
    query_date = datetime.date.today().isoformat()
    observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    source_name = "baostock.query_stock_industry"
    rows = []
    with DataFetcher._baostock_lock:
        socket.setdefaulttimeout(30)
        session = bs.login()
        if session.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {session.error_msg}")
        try:
            from baostock.common import context
            connection = getattr(context, "default_socket", None)
            if connection is not None:
                connection.settimeout(30)
            for target in targets:
                code = str(target.get("symbol") or "").zfill(6)
                exchange = str(target.get("ts_code") or "").split(".")[-1].lower()
                if len(code) != 6 or exchange not in ("sh", "sz", "bj"):
                    raise RuntimeError(f"身份代码格式异常: {target}")
                rs = bs.query_stock_industry(code=f"{exchange}.{code}", date=query_date)
                if rs is None or rs.error_code != "0":
                    raise RuntimeError(f"BaoStock 行业查询失败 {code}: {getattr(rs, 'error_msg', '无响应')}")
                matches = []
                while rs.next():
                    record = rs.get_row_data()
                    if len(record) >= 5 and record[1] == f"{exchange}.{code}":
                        matches.append(record)
                if len(matches) != 1:
                    raise RuntimeError(f"BaoStock 行业返回不唯一 {code}: {len(matches)}")
                record = matches[0]
                industry = str(record[3] or "").strip()
                classification = str(record[4] or "").strip()
                source_update_date = _date(record[0])
                if not industry or not source_update_date:
                    raise RuntimeError(f"BaoStock 行业记录缺字段 {code}: {record}")
                rows.append((source_name, observed_at, query_date, source_update_date,
                             target["ts_code"], record[2], industry, classification))
        finally:
            bs.logout()

    def _save(cursor):
        cursor.executemany("""INSERT INTO stock_identity_source_observations
            (source_name,observed_at,query_date,source_update_date,ts_code,code_name,industry,industry_classification)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""", rows)
        filled_count = 0
        verified_count = 0
        for row in rows:
            cursor.execute("""UPDATE stock_identity
                SET industry=%s,industry_source=%s,industry_classification=%s,industry_as_of=%s
                WHERE ts_code=%s AND ((industry IS NULL OR industry='') OR industry_source=%s)""",
                (row[6], row[0], row[7], row[3], row[4], row[0]))
            filled_count += max(0, cursor.rowcount)
            cursor.execute("""SELECT industry,industry_source,industry_classification,industry_as_of
                FROM stock_identity WHERE ts_code=%s""", (row[4],))
            current = cursor.fetchone()
            if current and (current["industry"], current["industry_source"],
                            current["industry_classification"], str(current["industry_as_of"])) == (
                                row[6], row[0], row[7], row[3]):
                verified_count += 1
        return filled_count, verified_count

    filled_count, verified_count = Database.transaction(_save)
    return {
        "status": "complete" if verified_count == len(rows) else "partial",
        "source": source_name,
        "query_date": query_date,
        "observed_at": observed_at.isoformat(),
        "target_count": len(targets),
        "observed_count": len(rows),
        "filled_count": filled_count,
        "verified_count": verified_count,
    }


def collect_stock_index_membership():
    from core.data import Database
    from core.data.sources.tushare import _rate_limit, get_pro

    ensure_stock_context_tables()
    pro = get_pro()
    if not pro:
        raise RuntimeError("TuShare 不可用")
    total = 0
    membership_by_key = {}
    page_fingerprints = set()
    for is_new in ("N", "Y"):
        offset = 0
        while True:
            _rate_limit()
            df = pro.query("index_member_all", is_new=is_new,
                           fields="ts_code,l1_code,l1_name,in_date,out_date,is_new",
                           limit=2000, offset=offset)
            if df is None:
                raise RuntimeError(f"TuShare index_member_all({is_new}) offset={offset} 返回 None")
            required_columns = {"ts_code", "l1_code", "l1_name", "in_date", "out_date", "is_new"}
            if not required_columns.issubset(df.columns):
                raise RuntimeError(f"TuShare index_member_all({is_new}) 缺少字段: {sorted(required_columns - set(df.columns))}")
            if df.empty:
                break
            page_keys = []
            for r in df.to_dict("records"):
                ts_code = r.get("ts_code")
                industry_code = r.get("l1_code")
                in_date = _date(r.get("in_date"))
                if not ts_code or not industry_code or not in_date:
                    raise RuntimeError(f"TuShare index_member_all({is_new}) 存在身份或生效日期缺失记录")
                row = (ts_code, industry_code, r.get("l1_name"), ts_code,
                       in_date, _date(r.get("out_date")), r.get("is_new"), datetime.date.today().isoformat())
                key = (ts_code, industry_code, in_date)
                existing = membership_by_key.get(key)
                if existing is not None and existing != row:
                    raise RuntimeError(f"TuShare index_member_all 成员版本冲突: {key}")
                membership_by_key[key] = row
                page_keys.append(key)
            page_fingerprint = tuple(page_keys)
            if page_fingerprint in page_fingerprints:
                raise RuntimeError(f"TuShare index_member_all({is_new}) offset={offset} 重复返回同一页")
            page_fingerprints.add(page_fingerprint)
            if len(df) < 2000:
                break
            offset += len(df)
    rows = list(membership_by_key.values())
    if len(rows) < 4000:
        raise RuntimeError(f"TuShare index_member_all 快照覆盖异常: {len(rows)} 行")
    observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    observations = [(observed_at, row[0], row[1], row[2], row[4], row[5], row[6]) for row in rows]
    def _save(cursor):
        cursor.executemany("""INSERT INTO stock_index_membership_observations
            (observed_at,ts_code,index_code,index_name,in_date,out_date,is_new)
            VALUES (%s,%s,%s,%s,%s,%s,%s)""", observations)
        cursor.executemany("""INSERT INTO stock_index_membership
            (ts_code,index_code,index_name,con_code,in_date,out_date,is_new,updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE
            index_name=VALUES(index_name),out_date=VALUES(out_date),is_new=VALUES(is_new),updated_at=VALUES(updated_at)""", rows)
        return len(rows)
    return Database.transaction(_save)


def collect_stock_concept_membership(concept_snapshot=None):
    from core.data.stock_sectors_migration import ensure_stock_sector_observations
    ensure_stock_sector_observations()

    def _collect_tushare():
        from core.data import Database
        from core.data.sources.tushare import _rate_limit, get_pro

        pro = get_pro()
        if not pro:
            raise RuntimeError("TuShare 不可用")

        index_df = None
        for retry in range(3):
            _rate_limit()
            try:
                index_df = pro.ths_index(exchange="A", type="N", fields="ts_code,name,count,exchange,list_date,type")
                break
            except Exception as exc:
                if retry == 2 or type(exc).__name__ not in ("ReadTimeout", "ConnectTimeout", "ConnectionError"):
                    raise
                time.sleep(3 * (retry + 1))
        required_index_columns = {"ts_code", "name", "type", "exchange"}
        if index_df is None or not required_index_columns.issubset(index_df.columns):
            raise RuntimeError("TuShare ths_index 未返回完整概念板块字段")
        concepts = [
            (str(row["ts_code"]), str(row["name"]), str(row.get("count") or ""))
            for row in index_df.to_dict("records")
            if str(row.get("type") or "") == "N"
            and str(row.get("exchange") or "") == "A"
            and row.get("ts_code")
            and row.get("name")
        ]
        if len(concepts) < 200 or len({code for code, _, _ in concepts}) != len(concepts):
            raise RuntimeError(f"TuShare 概念板块目录覆盖或唯一性异常: {len(concepts)} 个")

        active_codes = {
            str(row["symbol"]).zfill(6)
            for row in Database.fetchall(
                "SELECT symbol FROM stock_identity WHERE list_status='L' AND symbol IS NOT NULL"
            )
        }
        if len(active_codes) < 4000:
            raise RuntimeError(f"当前证券身份表覆盖不足，无法校验概念成分: {len(active_codes)} 只")

        observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        observations = []
        current_rows = set()
        seen_codes = set()
        for index, (sector_code, sector_name, expected_count) in enumerate(concepts, 1):
            members = None
            for retry in range(3):
                _rate_limit()
                try:
                    members = pro.ths_member(ts_code=sector_code, fields="ts_code,con_code,con_name")
                    break
                except Exception as exc:
                    if retry == 2 or type(exc).__name__ not in ("ReadTimeout", "ConnectTimeout", "ConnectionError"):
                        raise
                    time.sleep(3 * (retry + 1))
            if index % 50 == 0 or index == len(concepts):
                logger.info("TuShare 概念成员进度 %s/%s", index, len(concepts))
            required_member_columns = {"ts_code", "con_code", "con_name"}
            if members is None or members.empty or not required_member_columns.issubset(members.columns):
                raise RuntimeError(f"TuShare ths_member({sector_code}) 返回空数据或字段不全")
            board_codes = set()
            for row in members.to_dict("records"):
                if str(row.get("ts_code") or "") != sector_code:
                    raise RuntimeError(f"TuShare ths_member({sector_code}) 混入其他板块代码")
                raw_code = str(row.get("con_code") or "")
                code = raw_code.split(".")[0]
                if not code.isdigit() or len(code) != 6:
                    raise RuntimeError(f"TuShare ths_member({sector_code}) 存在非法股票代码: {raw_code}")
                board_codes.add(code)
            if not board_codes:
                raise RuntimeError(f"TuShare ths_member({sector_code}) 没有有效成员")
            if expected_count.isdigit() and len(board_codes) != int(expected_count):
                raise RuntimeError(
                    f"TuShare ths_member({sector_code}) 成员数 {len(board_codes)} "
                    f"与 ths_index 声明数 {expected_count} 不一致"
                )
            unknown = board_codes - active_codes
            if unknown:
                raise RuntimeError(f"TuShare ths_member({sector_code}) 有 {len(unknown)} 只股票未匹配当前上市身份")
            seen_codes.update(board_codes)
            observations.extend((observed_at, "tushare.ths_member", sector_code, sector_name, code)
                                for code in board_codes)
            current_rows.update((code, sector_name, "concept", observed_at) for code in board_codes)

        if len(observations) < 5000 or len(seen_codes) < 3000 or len(current_rows) < 5000:
            raise RuntimeError(
                f"TuShare 概念成员快照覆盖异常: {len(concepts)} 板块, "
                f"{len(observations)} 关系, {len(seen_codes)} 只证券"
            )

        observation_rows = list(dict.fromkeys(observations))
        current_rows = list(current_rows)

        def _save(cursor):
            cursor.executemany("""INSERT INTO stock_sector_membership_observations
                (observed_at,source,sector_code,sector_name,code) VALUES (%s,%s,%s,%s,%s)""",
                               observation_rows)
            cursor.execute("DELETE FROM stock_sectors WHERE sector_type='concept'")
            cursor.executemany("""INSERT INTO stock_sectors
                (code,sector_name,sector_type,updated_at) VALUES (%s,%s,%s,%s)""", current_rows)
            return len(current_rows)

        return {
            "status": "success",
            "source": "tushare.ths_member",
            "observed_at": observed_at.isoformat(),
            "concepts": len(concepts),
            "observed_relations": len(observation_rows),
            "current_relations": Database.transaction(_save),
        }

    try:
        return _collect_tushare()
    except Exception as primary_error:
        eastmoney_error = None
        try:
            from core.data import Database
            from core.data.sources.eastmoney_concepts import fetch_concept_memberships

            snapshot = concept_snapshot or fetch_concept_memberships()
            active_codes = {
                str(row["symbol"]).zfill(6)
                for row in Database.fetchall(
                    "SELECT symbol FROM stock_identity WHERE list_status='L' AND symbol IS NOT NULL"
                )
            }
            if len(active_codes) < 4000:
                raise RuntimeError(f"当前证券身份表覆盖不足，无法校验概念成分: {len(active_codes)} 只")

            observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            observation_rows = []
            current_rows = set()
            for sector_code, sector_name, members in snapshot["concepts"]:
                unknown = members - active_codes
                if unknown:
                    raise RuntimeError(
                        f"Eastmoney {sector_code} 有 {len(unknown)} 只成分未匹配当前上市身份"
                    )
                observation_rows.extend(
                    (observed_at, "eastmoney.concept_members", sector_code, sector_name, code)
                    for code in members
                )
                current_rows.update(
                    (code, sector_name, "concept", observed_at) for code in members
                )

            observation_rows = list(dict.fromkeys(observation_rows))
            current_rows = list(current_rows)

            def _save_eastmoney(cursor):
                cursor.executemany("""INSERT INTO stock_sector_membership_observations
                    (observed_at,source,sector_code,sector_name,code) VALUES (%s,%s,%s,%s,%s)""",
                                   observation_rows)
                cursor.execute("DELETE FROM stock_sectors WHERE sector_type='concept'")
                cursor.executemany("""INSERT INTO stock_sectors
                    (code,sector_name,sector_type,updated_at) VALUES (%s,%s,%s,%s)""", current_rows)
                return len(current_rows)

            return {
                "status": "success",
                "source": "eastmoney.concept_members",
                "fallback_from": f"tushare.ths_member: {str(primary_error)[:300]}",
                "observed_at": observed_at.isoformat(),
                "concepts": snapshot["concept_count"],
                "observed_relations": len(observation_rows),
                "current_relations": Database.transaction(_save_eastmoney),
            }
        except Exception as exc:
            eastmoney_error = exc
            logger.warning("东方财富概念采集失败，继续同花顺回退: %s", eastmoney_error)
        from core.shared.infrastructure.jobs import DataCollectionScheduler

        fallback = DataCollectionScheduler()._collect_concept_sectors()
        if fallback.get("status") != "success":
            raise RuntimeError(
                f"概念成分采集未完成；TuShare: {str(primary_error)[:300]}；"
                f"东方财富: {str(eastmoney_error)[:300]}；"
                f"同花顺: {fallback.get('error_message') or fallback.get('status')}"
            ) from primary_error
        return {
            "status": fallback["status"],
            "source": fallback.get("source_counts", {}),
            "observed_at": fallback.get("observed_at"),
            "concepts": fallback.get("concept_count", 0),
            "observed_relations": fallback.get("observed_relations", 0),
            "current_relations": fallback.get("total_inserted", 0),
        }


def collect_dividend_events(days=30):
    import hashlib
    from core.data import Database
    from core.data.repos.dividend_source_versions import save_dividend_source_versions
    from core.data.sources.tushare import _rate_limit, get_pro

    ensure_stock_context_tables()
    pro = get_pro()
    if not pro:
        raise RuntimeError("TuShare 不可用")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=max(7, int(days)))
    observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    capture_id = hashlib.sha256(f"stock_dividend_events|{observed_at.isoformat()}".encode("utf-8")).hexdigest()
    source_row_no = 0
    total = 0
    for offset in range((end - start).days + 1):
        trade_day = (start + datetime.timedelta(days=offset)).strftime("%Y%m%d")
        _rate_limit()
        frame = pro.dividend(ann_date=str(trade_day), fields="ts_code,end_date,ann_date,div_proc,stk_div,cash_div,cash_div_tax,record_date,ex_date,pay_date,imp_ann_date")
        if frame is None or frame.empty:
            continue
        if len(frame) >= 2000:
            raise RuntimeError(f"分红公告日 {trade_day} 返回达到 TuShare 2000 行上限，需人工分页核验")
        source_records = []
        for raw_record in frame.to_dict("records"):
            record = {}
            for key, value in raw_record.items():
                if value is None or str(value) in ("", "nan", "NaT"):
                    value = None
                elif hasattr(value, "item"):
                    value = value.item()
                record[key] = value
            if record.get("ts_code") and record.get("end_date"):
                source_records.append(record)
        save_dividend_source_versions(capture_id, observed_at, source_records, source_row_no)
        source_row_no += len(source_records)
        grouped = {}
        for record in source_records:
            key = (record.get("ts_code"), record.get("end_date"), record.get("ann_date"), record.get("div_proc"))
            grouped.setdefault(key, []).append(record)
        conflict_keys = []
        for key, records in grouped.items():
            payloads = {str(sorted(record.items())) for record in records}
            if len(payloads) > 1:
                conflict_keys.append(key)
        rows = [(
            r.get("ts_code"), _date(r.get("end_date")), _date(r.get("ann_date")), r.get("div_proc"),
            r.get("stk_div"), r.get("cash_div"), r.get("cash_div_tax"), _date(r.get("record_date")),
            _date(r.get("ex_date")), _date(r.get("pay_date")), _date(r.get("imp_ann_date")),
        ) for key, records in grouped.items() if key not in conflict_keys for r in records[:1]]
        Database.executemany("""INSERT INTO stock_dividend_events
            (ts_code,end_date,ann_date,div_proc,stk_div,cash_div,cash_div_tax,record_date,ex_date,pay_date,imp_ann_date,data_quality_status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'source_consistent_snapshot') ON DUPLICATE KEY UPDATE
            ann_date=VALUES(ann_date),div_proc=VALUES(div_proc),stk_div=VALUES(stk_div),cash_div=VALUES(cash_div),
            cash_div_tax=VALUES(cash_div_tax),record_date=VALUES(record_date),ex_date=VALUES(ex_date),
            pay_date=VALUES(pay_date),imp_ann_date=VALUES(imp_ann_date),data_quality_status=VALUES(data_quality_status)""", rows)
        if conflict_keys:
            Database.executemany("""UPDATE stock_dividend_events SET data_quality_status='source_conflict'
                WHERE ts_code=%s AND end_date=%s AND ann_date <=> %s AND div_proc <=> %s""", [
                (code, _date(end_date), _date(ann_date), div_proc)
                for code, end_date, ann_date, div_proc in conflict_keys
            ])
        total += len(rows)
    return total
