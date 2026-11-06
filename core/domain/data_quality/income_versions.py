from datetime import date
from decimal import Decimal


_FINANCIAL_FIELDS = ("operate_profit", "total_revenue", "oper_cost", "sell_exp", "admin_exp", "fin_exp")
_VERSION_FIELDS = ("ann_date", "f_ann_date", "update_flag", *_FINANCIAL_FIELDS)


def _number(value):
    return None if value is None else Decimal(str(value))


def _date(value):
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).replace("-", "")[:8]
    if len(text) != 8 or not text.isdigit():
        return None
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _financial_signature(row):
    return tuple(_number(row.get(field)) for field in _FINANCIAL_FIELDS), str(row.get("ann_date") or "")


def _matches(left, right, fields):
    for field in fields:
        if field in _FINANCIAL_FIELDS:
            if _number(left.get(field)) != _number(right.get(field)):
                return False
        elif field in ("ann_date", "f_ann_date"):
            if _date(left.get(field)) != _date(right.get(field)):
                return False
        elif str(left.get(field) or "") != str(right.get(field) or ""):
            return False
    return True


def classify_income_version_candidates(candidate_rows, source_by_key, current_by_key):
    actions = []
    ledger = []
    for candidate in candidate_rows:
        code = str(candidate["code"])
        report_date = str(candidate["report_date"])[:10]
        key = (code, report_date)
        records = [
            row for row in source_by_key.get(key, [])
            if str(row.get("report_type")) == "1"
            and str(row.get("comp_type")) == "1"
            and row.get("f_ann_date")
            and row.get("total_revenue") is not None
            and row.get("oper_cost") is not None
        ]
        current = current_by_key.get(key)
        if not records:
            ledger.append({"code": code, "report_date": report_date, "status": "source_missing"})
            continue
        if not current:
            ledger.append({"code": code, "report_date": report_date, "status": "current_row_missing"})
            continue
        latest_date = max(str(row["f_ann_date"]).replace("-", "")[:8] for row in records)
        latest_rows = [row for row in records if str(row["f_ann_date"]).replace("-", "")[:8] == latest_date]
        signatures = {_financial_signature(row) for row in latest_rows}
        if len(signatures) != 1:
            ledger.append({"code": code, "report_date": report_date, "status": "source_conflict", "latest_f_ann_date": latest_date})
            continue
        source_latest = max(latest_rows, key=lambda row: str(row.get("update_flag") or ""))
        latest = dict(source_latest)
        latest["ann_date"] = _date(latest.get("ann_date"))
        latest["f_ann_date"] = _date(latest.get("f_ann_date"))
        latest["update_flag"] = str(latest.get("update_flag") or "")
        current_pair = (_number(current.get("total_revenue")), _number(current.get("oper_cost")))
        latest_pair = (_number(latest.get("total_revenue")), _number(latest.get("oper_cost")))
        source_pairs = {
            (str(row["f_ann_date"]).replace("-", "")[:8], _number(row.get("total_revenue")), _number(row.get("oper_cost")))
            for row in records
        }
        if current_pair == latest_pair:
            current_version = {
                **current,
                "f_ann_date": _date(current.get("f_ann_date")),
                "ann_date": _date(current.get("ann_date")),
            }
            status = "already_current" if _matches(current_version, latest, _VERSION_FIELDS) else "version_aligned"
        else:
            matching_versions = {
                fann for fann, revenue, cost in source_pairs
                if revenue == current_pair[0] and cost == current_pair[1]
            }
            if matching_versions:
                status = "stale_version"
            else:
                revenue_versions = {fann for fann, revenue, _ in source_pairs if revenue == current_pair[0]}
                cost_versions = {fann for fann, _, cost in source_pairs if cost == current_pair[1]}
                status = "mixed_version" if revenue_versions and cost_versions and not revenue_versions.intersection(cost_versions) else "source_values_unmatched"
        entry = {"code": code, "report_date": report_date, "status": status, "latest_f_ann_date": latest_date}
        ledger.append(entry)
        if status != "already_current" and status in ("version_aligned", "stale_version", "mixed_version"):
            selected = {field: latest.get(field) for field in _VERSION_FIELDS}
            actions.append({
                "code": code,
                "end_date": report_date,
                "expected": current,
                "selected": selected,
                "source_rows": records,
                "status": status,
            })
    return actions, ledger
