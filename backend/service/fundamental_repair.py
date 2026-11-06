import datetime
from decimal import Decimal

from core.data.repos.fundamental_versions import (
    apply_official_fundamental_field_repairs,
    apply_official_gpm_repairs,
    ensure_fundamental_versions_table,
    get_fundamental_field_repair_rows,
    get_fundamental_gpm_repair_rows,
)
from core.shared.persistence import load_bytes, load_json, save_json_atomic


def repair_officially_verified_grossprofit_margins(evidence_path="output/gpm_official_evidence_20261025.json"):
    evidence_bytes = load_bytes(evidence_path)
    evidence = load_json(evidence_path)
    if not evidence_bytes or not evidence:
        raise RuntimeError("官方毛利率证据文件不存在")
    evidence_rows = evidence.get("items") or evidence.get("rows") or []
    keys = [(row["code"], row["report_date"]) for row in evidence_rows]
    current_rows = get_fundamental_gpm_repair_rows(keys)
    current_by_key = {(row["code"], str(row["report_date"])): row for row in current_rows}
    if set(current_by_key) != set(keys):
        raise RuntimeError("正式基本面行覆盖不完整")
    actions = []
    backup_rows = []
    for item in evidence_rows:
        key = (item["code"], item["report_date"])
        selected_gpm = (Decimal(str(item["official_revenue"])) - Decimal(str(item["official_oper_cost"])))
        selected_gpm = (selected_gpm / Decimal(str(item["official_revenue"])) * Decimal("100")).quantize(Decimal("0.0001"))
        if selected_gpm != Decimal(str(item["expected_gpm"])):
            raise RuntimeError(f"官方报告公式与登记毛利率不一致: {key[0]}/{key[1]}")
        current = current_by_key[key]
        if item.get("ann_date") and str(current.get("ann_date")) != item["ann_date"]:
            raise RuntimeError(f"正式基本面公告日与证据不一致: {key[0]}/{key[1]}")
        expected_current = {
            "ann_date": current["ann_date"],
            "grossprofit_margin": current["grossprofit_margin"],
        }
        backup_rows.append({
            "code": item["code"], "report_date": item["report_date"],
            "before": {field: str(value) if value is not None else None for field, value in expected_current.items()},
            "selected_gpm": str(selected_gpm),
        })
        if current["grossprofit_margin"] is not None and Decimal(str(current["grossprofit_margin"])) == selected_gpm:
            continue
        source_observed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        actions.append({
            "code": item["code"], "report_date": item["report_date"],
            "ann_date": current["ann_date"], "expected": expected_current,
            "selected_gpm": selected_gpm,
            "source_capture_id": f"official-gpm-{item['code']}-{item['report_date'].replace('-', '')}",
            "source_observed_at": source_observed_at,
            "evidence_record": {
                "ts_code": f"{item['code']}.{item.get('suffix') or ('SH' if item['code'].startswith('6') else 'BJ' if item['code'].startswith(('4', '8', '9')) else 'SZ')}",
                "end_date": item["report_date"].replace("-", ""),
                "ann_date": item["ann_date"].replace("-", ""),
                "update_flag": item.get("source_update_flag"),
                "grossprofit_margin": float(selected_gpm),
                "official_revenue": item["official_revenue"],
                "official_oper_cost": item["official_oper_cost"],
                "report_url": item["report_url"],
                "source_name": "official_report",
                "official_available_date": evidence.get("official_available_date"),
                "basis": "(official_revenue - official_oper_cost) / official_revenue * 100",
            },
        })
    backup_path = evidence.get("backup_path") or evidence_path.replace(".json", "_prewrite.json")
    backup = {
        "evidence_sha256": __import__("hashlib").sha256(evidence_bytes).hexdigest(),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": backup_rows,
    }
    existing = load_json(backup_path)
    if existing:
        if existing.get("evidence_sha256") != backup["evidence_sha256"]:
            raise RuntimeError("已有毛利率备份对应不同证据文件")
        saved = {(row["code"], row["report_date"]): row for row in existing.get("rows", [])}
        for row in backup_rows:
            old = saved.get((row["code"], row["report_date"]))
            if not old:
                raise RuntimeError(f"毛利率备份缺少目标行: {row['code']}/{row['report_date']}")
            current_value = row["before"]["grossprofit_margin"]
            if current_value not in (old["before"]["grossprofit_margin"], old["selected_gpm"]):
                raise RuntimeError(f"正式毛利率偏离备份及已选证据，停止修复: {row['code']}/{row['report_date']}")
    else:
        save_json_atomic(backup, backup_path)
    applied = apply_official_gpm_repairs(actions) if actions else []
    final_rows = get_fundamental_gpm_repair_rows(keys)
    final_by_key = {(row["code"], str(row["report_date"])): row for row in final_rows}
    for item in evidence_rows:
        actual = Decimal(str(final_by_key[(item["code"], item["report_date"])] ["grossprofit_margin"]))
        if actual != Decimal(str(item["expected_gpm"])):
            raise RuntimeError(f"毛利率正式值读回不符: {item['code']}/{item['report_date']}")
    return {"rows_verified": len(evidence_rows), "rows_written": len(applied),
            "applied": applied, "backup_path": backup_path}


def repair_officially_verified_fundamental_fields(evidence_path):
    ensure_fundamental_versions_table()
    evidence_bytes = load_bytes(evidence_path)
    evidence = load_json(evidence_path)
    if not evidence_bytes or not evidence or not evidence.get("rows"):
        raise RuntimeError("官方基本面字段证据文件不存在或为空")
    rows = evidence["rows"]
    keys = [(item["code"], item["report_date"]) for item in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("官方基本面字段证据包含重复证券/报告期")
    fields = sorted({field for item in rows for field in item.get("selected", {})})
    current_rows = get_fundamental_field_repair_rows(keys, fields)
    current_by_key = {(row["code"], str(row["report_date"])): row for row in current_rows}
    if set(current_by_key) != set(keys):
        raise RuntimeError("正式基本面行覆盖不完整")
    actions = []
    backup_rows = []
    def _same(left, right):
        if left is None or right is None:
            return left is right
        if isinstance(left, (datetime.date, datetime.datetime)) or isinstance(right, (datetime.date, datetime.datetime)):
            return str(left)[:10] == str(right)[:10]
        try:
            return Decimal(str(left)) == Decimal(str(right))
        except Exception:
            return left == right

    for item in rows:
        key = (item["code"], item["report_date"])
        current = current_by_key[key]
        if str(current.get("ann_date")) != item["ann_date"]:
            raise RuntimeError(f"正式基本面公告日与官方证据不匹配: {key[0]}/{key[1]}")
        expected = {"ann_date": current["ann_date"]}
        expected.update({field: current.get(field) for field in item["selected"]})
        supplied = item.get("expected") or {}
        for field, value in supplied.items():
            if not _same(current.get(field), value) and not _same(current.get(field), item["selected"].get(field)):
                raise RuntimeError(f"正式基本面行偏离写前值及官方选值: {key[0]}/{key[1]}/{field}")
        selected = {field: Decimal(str(value)) if value is not None else None
                    for field, value in item["selected"].items()}
        if not selected:
            raise RuntimeError(f"官方基本面字段为空: {key[0]}/{key[1]}")
        backup_rows.append({
            "code": item["code"], "report_date": item["report_date"],
            "before": {field: str(current.get(field)) if current.get(field) is not None else None
                       for field in expected},
            "selected": {field: str(value) if value is not None else None
                         for field, value in selected.items()},
        })
        record = {
            "ts_code": f"{item['code']}.{item.get('suffix') or ('SH' if item['code'].startswith('6') else 'BJ' if item['code'].startswith(('4', '8', '9')) else 'SZ')}",
            "end_date": item["report_date"].replace("-", ""),
            "ann_date": item["ann_date"].replace("-", ""),
            "update_flag": item.get("source_update_flag"),
            "official_available_date": item.get("official_available_date"),
            "official_values": {field: str(value) if value is not None else None
                                for field, value in selected.items()},
            "official_classification": item.get("official_classification") or {},
            "report_url": item["report_url"],
            "source_name": "official_report",
        }
        actions.append({
            "code": item["code"], "report_date": item["report_date"],
            "ann_date": current["ann_date"], "expected": expected,
            "selected": selected,
            "source_capture_id": item.get("source_capture_id") or f"official-fields-{item['code']}-{item['report_date'].replace('-', '')}",
            "source_observed_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            "evidence_record": record,
        })
    backup_path = evidence.get("backup_path") or evidence_path.replace(".json", "_prewrite.json")
    backup = {
        "evidence_sha256": __import__("hashlib").sha256(evidence_bytes).hexdigest(),
        "official_report_url": evidence["official_report_url"],
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": backup_rows,
    }
    existing = load_json(backup_path)
    if existing:
        if existing.get("evidence_sha256") != backup["evidence_sha256"]:
            raise RuntimeError("已有官方基本面字段备份对应不同证据文件")
        saved = {(row["code"], row["report_date"]): row for row in existing.get("rows", [])}
        for row in backup_rows:
            prior = saved.get((row["code"], row["report_date"]))
            if not prior or any(
                not _same(value, prior.get("before", {}).get(field))
                and not _same(value, prior.get("selected", {}).get(field))
                for field, value in row["before"].items() if field != "ann_date"
            ) or not _same(row["before"].get("ann_date"), prior.get("before", {}).get("ann_date")):
                raise RuntimeError(f"正式基本面行偏离备份及已选证据: {row['code']}/{row['report_date']}")
    else:
        save_json_atomic(backup, backup_path)
    applied = apply_official_fundamental_field_repairs(actions)
    final_rows = get_fundamental_field_repair_rows(keys, fields)
    final_by_key = {(row["code"], str(row["report_date"])): row for row in final_rows}
    for item in rows:
        actual = final_by_key[(item["code"], item["report_date"])]
        if any(not _same(actual.get(field), value)
               for field, value in item["selected"].items()):
            raise RuntimeError(f"官方基本面字段正式值读回不符: {item['code']}/{item['report_date']}")
    return {"rows_verified": len(rows), "rows_written": sum(row["status"] == "repaired" for row in applied),
            "applied": applied, "backup_path": backup_path, "evidence_path": evidence_path}
