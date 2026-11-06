import datetime
import json
import logging
import math
import uuid
from collections import defaultdict

from core.data import Database
from core.data.repos.fundamental_versions import ensure_fundamental_versions_table, save_fundamental_batch
from core.shared.fundamental_contract import FUNDAMENTAL_CURRENT_PROJECTION, FUNDAMENTAL_SOURCE_FIELDS

logger = logging.getLogger(__name__)


def fetch_and_save_fundamentals(codes=None, max_workers=4):
    if not Database.is_available():
        return {"saved": 0, "tushare": 0, "akshare": 0, "versions_saved": 0}

    if codes is None:
        from core.data.sources.tushare import get_all_stock_codes
        codes = get_all_stock_codes()
    codes = list(codes or [])
    if not codes:
        return {"saved": 0, "tushare": 0, "akshare": 0, "versions_saved": 0}

    identity_rows = Database.fetchall("SELECT ts_code FROM stock_identity")
    ts_code_by_code = {
        str(row["ts_code"]).split(".")[0]: str(row["ts_code"])
        for row in identity_rows if row.get("ts_code")
    }
    existing_rows = Database.fetchall("SELECT DISTINCT code FROM fundamental_data")
    existing_codes = {str(row["code"]) for row in existing_rows}

    from core.data.sources.tushare import _rate_limit, get_pro
    pro = get_pro()
    if not pro:
        return {"saved": 0, "tushare": 0, "akshare": 0, "versions_saved": 0}

    ensure_fundamental_versions_table()
    fields = ",".join(("ts_code", "ann_date", "end_date", "update_flag", *FUNDAMENTAL_SOURCE_FIELDS))

    def _safe_float(value):
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _date(value):
        if value is None:
            return None
        value = str(value).replace("-", "")[:8]
        try:
            return datetime.datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            return None

    capture_id = uuid.uuid4().hex
    observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    saved = 0
    versions_saved = 0
    failed_batches = 0
    failed_codes = []
    conflicted_codes = []
    total_conflicted_periods = 0
    end_date = datetime.date.today().strftime("%Y%m%d")

    def _update_flag(row):
        value = row.get("update_flag")
        try:
            return (1, int(value))
        except (TypeError, ValueError):
            return (0, str(value or ""))

    for start in range(len(codes)):
        code = codes[start]
        code_text = str(code)
        bare_code = code_text.split(".")[0]
        if "." in code_text:
            ts_code = code_text
        else:
            ts_code = ts_code_by_code.get(bare_code)
            if not ts_code:
                suffix = "SH" if bare_code.startswith("6") else "BJ" if bare_code.startswith(("4", "8", "9")) else "SZ"
                ts_code = f"{bare_code}.{suffix}"
        try:
            _rate_limit()
            frame = pro.fina_indicator(
                ts_code=ts_code,
                start_date="20180101",
                end_date=end_date,
                fields=fields,
                limit=100,
            )
            response_frames = []
            if frame is not None and len(frame) < 100:
                response_frames.append(frame)
            elif frame is not None:
                today = datetime.date.today()
                for year in range(2018, today.year + 1):
                    period_start = datetime.date(year, 1, 1)
                    period_end = min(datetime.date(year, 12, 31), today)
                    _rate_limit()
                    year_frame = pro.fina_indicator(
                        ts_code=ts_code,
                        start_date=period_start.strftime("%Y%m%d"),
                        end_date=period_end.strftime("%Y%m%d"),
                        fields=fields,
                        limit=100,
                    )
                    if year_frame is None:
                        continue
                    if len(year_frame) < 100:
                        response_frames.append(year_frame)
                        continue
                    quarter_ranges = (
                        (datetime.date(year, 1, 1), datetime.date(year, 3, 31)),
                        (datetime.date(year, 4, 1), datetime.date(year, 6, 30)),
                        (datetime.date(year, 7, 1), datetime.date(year, 9, 30)),
                        (datetime.date(year, 10, 1), period_end),
                    )
                    for quarter_start, quarter_end in quarter_ranges:
                        if quarter_start > period_end:
                            continue
                        _rate_limit()
                        quarter_frame = pro.fina_indicator(
                            ts_code=ts_code,
                            start_date=quarter_start.strftime("%Y%m%d"),
                            end_date=min(quarter_end, period_end).strftime("%Y%m%d"),
                            fields=fields,
                            limit=100,
                        )
                        if quarter_frame is not None and len(quarter_frame) >= 100:
                            raise RuntimeError(f"TuShare 财务指标季度分段仍达到100行上限: {year}/{quarter_start.month}")
                        if quarter_frame is not None:
                            response_frames.append(quarter_frame)
            source_records = []
            for response_frame in response_frames:
                if response_frame is not None and not response_frame.empty:
                    source_records.extend(response_frame.astype(object).where(response_frame.notna(), None).to_dict("records"))
            if not source_records and bare_code in existing_codes:
                raise RuntimeError(f"已有基本面记录但 TuShare 返回空结果: {ts_code}")
            if not source_records:
                continue
            for row in source_records:
                response_code = str(row.get("ts_code") or "").strip().upper()
                if response_code != ts_code.upper():
                    raise RuntimeError(f"TuShare 返回证券身份与请求不一致: {response_code or 'missing'} != {ts_code}")
                if _date(row.get("end_date")) is None:
                    raise RuntimeError(f"TuShare 返回无效报告期: {row.get('end_date')}")
                if _date(row.get("ann_date")) is None:
                    raise RuntimeError(f"TuShare 财务指标缺少有效公告日: {row.get('ann_date')}")
            current_rows = []
            conflicted_periods = 0
            by_code_period = defaultdict(list)
            for row in source_records:
                record_code = str(row.get("ts_code") or "").split(".")[0]
                report_date = _date(row.get("end_date"))
                ann_date = _date(row.get("ann_date"))
                by_code_period[(record_code, report_date)].append((ann_date, row))
            for (record_code, report_date), versions in by_code_period.items():
                latest_ann_date = max(item[0] for item in versions)
                latest_versions = [row for ann_date, row in versions if ann_date == latest_ann_date]
                fingerprints = {
                    json.dumps({key: value for key, value in row.items() if key != "update_flag"},
                               ensure_ascii=False, sort_keys=True, default=str,
                               separators=(",", ":"), allow_nan=False)
                    for row in latest_versions
                }
                if len(fingerprints) != 1:
                    conflicted_periods += 1
                    continue
                row = max(latest_versions, key=_update_flag)
                ann_date = _date(row.get("ann_date"))
                current_rows.append((
                    record_code, report_date, ann_date,
                    *(_safe_float(row.get(source_field)) for source_field, _ in FUNDAMENTAL_CURRENT_PROJECTION),
                ))
            result = save_fundamental_batch(capture_id, observed_at, source_records, current_rows)
            saved += result["current_rows"]
            versions_saved += result["version_rows"]
            total_conflicted_periods += conflicted_periods
            if conflicted_periods:
                conflicted_codes.append(str(code))
        except Exception as error:
            failed_batches += 1
            failed_codes.append(str(code))
            logger.warning("[基本面] TuShare 批次 %s 失败: %s", start + 1, error)

    logger.info("[基本面] PIT 采集完成: %s 条；版本观察 %s 条；失败批次 %s；缺公告日行跳过当前表", saved, versions_saved, failed_batches)
    return {
        "saved": saved,
        "tushare": saved,
        "akshare": 0,
        "versions_saved": versions_saved,
        "capture_id": capture_id,
        "failed_batches": failed_batches,
        "failed_codes": failed_codes,
        "conflicted_codes": conflicted_codes,
        "conflicted_periods": total_conflicted_periods,
    }
