import datetime
import uuid

from core.data import Database
from core.data.repos.analyst_forecast_reports import (
    ensure_analyst_forecast_reports_tables,
    save_analyst_forecast_capture,
)


def collect_latest_analyst_forecast_reports():
    ensure_analyst_forecast_reports_tables()
    report_date = datetime.date.today() - datetime.timedelta(days=1)
    from core.shared.trading_time import get_latest_trading_day
    report_date = get_latest_trading_day(report_date)
    existing = Database.fetchone(
        "SELECT status,row_count FROM analyst_forecast_report_captures "
        "WHERE report_date=%s ORDER BY captured_at DESC LIMIT 1",
        (report_date,),
    )
    if existing and existing.get("status") in ("success", "empty"):
        return {"status": "already_captured", "report_date": report_date.isoformat(),
                "row_count": int(existing.get("row_count") or 0)}

    capture_id = uuid.uuid4().hex
    captured_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    rows = []
    status = "failed"
    try:
        from core.data.sources.tushare import _rate_limit, get_pro
        pro = get_pro()
        if not pro:
            raise RuntimeError("TuShare 客户端不可用")
        _rate_limit()
        frame = pro.report_rc(report_date=report_date.strftime("%Y%m%d"))
        if frame is not None and not frame.empty:
            if len(frame) >= 3000:
                status = "truncated"
                raise RuntimeError("report_rc 返回达到 3000 行上限，拒绝发布可能截断的日期快照")
            required = {"ts_code", "report_date", "quarter", "org_name", "eps", "np"}
            missing = required - set(frame.columns)
            if missing:
                raise RuntimeError(f"report_rc 缺少必要字段: {sorted(missing)}")
            rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
            for row in rows:
                source_date = str(row.get("report_date") or "").replace("-", "")[:8]
                if source_date != report_date.strftime("%Y%m%d"):
                    raise RuntimeError("report_rc 返回了与请求日期不一致的记录")
                source_time = row.get("create_time")
                if source_time is not None and hasattr(source_time, "to_pydatetime"):
                    row["create_time"] = source_time.to_pydatetime()
            status = "success"
        else:
            status = "empty"
        result = save_analyst_forecast_capture(capture_id, report_date, captured_at, rows, status)
        return {"status": status, "report_date": report_date.isoformat(), **result}
    except Exception as error:
        save_analyst_forecast_capture(capture_id, report_date, captured_at, [], status, str(error))
        return {"status": status, "report_date": report_date.isoformat(), "error": str(error)}
