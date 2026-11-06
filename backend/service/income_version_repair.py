import datetime
import json
import logging
from decimal import Decimal

from core.data.repos.income_versions import (
    apply_income_version_repairs,
    get_income_version_repair_rows,
    save_income_source_versions,
)
from core.shared.persistence import load_json, load_bytes, save_json_atomic

logger = logging.getLogger(__name__)


def collect_income_candidate_versions():
    candidate_path = "output/data_audit_live_20261025_candidates.json"
    output_path = "output/income_candidate_versions_shadow_20261025.json"
    candidate_bytes = load_bytes(candidate_path)
    candidate_data = load_json(candidate_path)
    if not candidate_bytes or not candidate_data:
        raise RuntimeError("财务候选清单或来源文件不存在")
    candidate_rows = candidate_data.get("margin_mismatch_candidates") or []
    candidate_keys = {(str(row["code"]), str(row["report_date"]).replace("-", "")) for row in candidate_rows}
    codes = sorted({code for code, _ in candidate_keys})
    if not codes:
        raise RuntimeError("财务候选清单为空")
    from core.data.sources.tushare import _rate_limit, get_pro
    pro = get_pro()
    if not pro:
        raise RuntimeError("TuShare 客户端不可用")
    source_hash = __import__("hashlib").sha256(candidate_bytes).hexdigest()
    state = load_json(output_path) or {}
    if state and state.get("candidate_sha256") != source_hash:
        raise RuntimeError("已有来源影子批次对应不同候选文件")
    state.update({
        "candidate_sha256": source_hash,
        "captured_at_utc": state.get("captured_at_utc") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "codes_total": len(codes),
        "codes": state.get("codes") or {},
    })
    fields = "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,update_flag,operate_profit,total_revenue,revenue,oper_cost,sell_exp,admin_exp,fin_exp"
    request_end = datetime.date.today().strftime("%Y%m%d")
    for index, code in enumerate(codes, 1):
        prior = state["codes"].get(code)
        if prior and prior.get("status") == "complete":
            continue
        suffix = "SH" if code.startswith("6") else "BJ" if code.startswith(("4", "8", "9")) else "SZ"
        try:
            _rate_limit()
            frame = pro.income(
                ts_code=f"{code}.{suffix}",
                start_date="20180101",
                end_date=request_end,
                fields=fields,
                limit=100,
            )
            if frame is None:
                state["codes"][code] = {"status": "empty", "records": [], "candidate_periods_found": []}
            elif len(frame) >= 100:
                state["codes"][code] = {"status": "truncated", "row_count": int(len(frame)), "records": []}
            else:
                records = json.loads(frame.to_json(orient="records", date_format="iso")) if not frame.empty else []
                selected = [row for row in records
                            if (str(row.get("ts_code") or "").split(".")[0], str(row.get("end_date") or "").replace("-", "")[:8]) in candidate_keys]
                found_periods = sorted({str(row.get("end_date") or "").replace("-", "")[:8] for row in selected})
                state["codes"][code] = {
                    "status": "complete",
                    "row_count": int(len(frame)),
                    "records": selected,
                    "candidate_periods_found": found_periods,
                }
        except Exception as error:
            state["codes"][code] = {"status": "failed", "error": f"{type(error).__name__}: {str(error)[:240]}", "records": []}
        save_json_atomic(state, output_path)
        if index % 25 == 0 or index == len(codes):
            complete = sum(item.get("status") == "complete" for item in state["codes"].values())
            logger.info("[财务候选源采集] %s/%s 证券完成 %s", index, len(codes), complete)
    source_records = [record for item in state["codes"].values() if item.get("status") == "complete" for record in item.get("records", [])]
    inserted = save_income_source_versions(source_records)
    found_keys = {
        (str(record.get("ts_code") or "").split(".")[0], str(record.get("end_date") or "").replace("-", "")[:8])
        for record in source_records
    }
    state["source_variants_inserted"] = inserted
    state["candidate_keys_total"] = len(candidate_keys)
    state["candidate_keys_with_source"] = len(candidate_keys & found_keys)
    state["candidate_keys_without_source"] = sorted([f"{code}/{period}" for code, period in candidate_keys - found_keys])
    state["codes_failed"] = sorted(code for code, item in state["codes"].items() if item.get("status") in ("failed", "truncated"))
    save_json_atomic(state, output_path)
    return {
        "codes_total": len(codes),
        "codes_complete": sum(item.get("status") == "complete" for item in state["codes"].values()),
        "codes_failed_or_truncated": state["codes_failed"],
        "candidate_keys_total": len(candidate_keys),
        "candidate_keys_with_source": state["candidate_keys_with_source"],
        "source_variants_inserted": inserted,
        "output_path": output_path,
    }


def repair_confirmed_income_version_conflicts():
    from core.data.income_data_migration import ensure_income_data_version_fields
    ensure_income_data_version_fields()
    source_path = "output/data_audit_live_20261025_source_samples.json"
    source_records = load_json(source_path)
    source_bytes = load_bytes(source_path)
    if not source_records or not source_bytes:
        raise RuntimeError("确认样例来源文件不存在")
    targets = {
        ("600361", "2020-12-31"),
        ("600822", "2023-12-31"),
        ("300448", "2025-06-30"),
    }
    current_rows = get_income_version_repair_rows(sorted(targets))
    current_by_key = {(row["code"], str(row["end_date"])): row for row in current_rows}
    repairs = []
    for code, end_date in sorted(targets):
        key = (code, end_date.replace("-", ""))
        records = source_records.get(f"{code}.{'SH' if code.startswith('6') else 'SZ'}", [])
        records = [row for row in records if str(row.get("end_date")) == key[1]
                   and str(row.get("report_type")) == "1" and str(row.get("comp_type")) == "1"]
        if len(records) < 2:
            raise RuntimeError(f"版本源记录不足: {code}/{end_date}")
        available = [row for row in records if row.get("f_ann_date") and row.get("total_revenue") is not None and row.get("oper_cost") is not None]
        latest_date = max(str(row["f_ann_date"]) for row in available)
        latest = [row for row in available if str(row["f_ann_date"]) == latest_date]
        values = {(row.get("total_revenue"), row.get("oper_cost"), str(row.get("update_flag") or "")) for row in latest}
        if len(values) != 1:
            raise RuntimeError(f"同一实际公告日存在冲突: {code}/{end_date}/{latest_date}")
        selected = latest[0]
        current = current_by_key.get((code, end_date))
        if not current:
            raise RuntimeError(f"正式利润表行不存在: {code}/{end_date}")
        repairs.append({
            "code": code,
            "end_date": end_date,
            "expected": current,
            "selected": {
                "total_revenue": selected["total_revenue"],
                "oper_cost": selected["oper_cost"],
                "f_ann_date": datetime.datetime.strptime(latest_date, "%Y%m%d").date(),
                "update_flag": str(selected.get("update_flag") or ""),
            },
            "source_rows": records,
        })
    if len(repairs) != len(targets):
        raise RuntimeError("样例行数量和目标不一致")
    backup = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_path": source_path,
        "source_sha256": __import__("hashlib").sha256(source_bytes).hexdigest(),
        "rows": [
            {"code": row["code"], "end_date": str(row["end_date"]),
             **{field: str(row[field]) if isinstance(row.get(field), (datetime.date, Decimal)) else row.get(field)
                for field in ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost")}}
            for row in current_rows
        ],
        "selected_source_rows": [item["selected"] for item in repairs],
    }
    backup = json.loads(json.dumps(backup, ensure_ascii=False, default=str))
    backup_path = "output/data_quality_repair_20261025_prewrite.json"
    existing_backup = load_json(backup_path)
    if existing_backup:
        if existing_backup.get("source_sha256") != backup["source_sha256"]:
            raise RuntimeError("已存在的修复备份与当前来源文件不符，停止重跑")
        prior = {(row["code"], row["end_date"]): row for row in existing_backup.get("rows", [])}
        for item in repairs:
            row = item["expected"]
            before = prior.get((item["code"], item["end_date"]))
            if not before:
                raise RuntimeError(f"备份中缺少原始行: {item['code']}/{item['end_date']}")
            normalize = lambda value: str(Decimal(str(value)).normalize()) if value is not None else None
            current_values = (str(row.get("f_ann_date")), normalize(row.get("total_revenue")), normalize(row.get("oper_cost")))
            before_values = (str(before.get("f_ann_date")), normalize(before.get("total_revenue")), normalize(before.get("oper_cost")))
            selected_values = (str(item["selected"]["f_ann_date"]),
                               normalize(item["selected"]["total_revenue"]), normalize(item["selected"]["oper_cost"]))
            if current_values not in (before_values, selected_values):
                raise RuntimeError(f"正式数据偏离备份和已选来源，停止重跑: {item['code']}/{item['end_date']}")
    else:
        save_json_atomic(backup, backup_path)
    return apply_income_version_repairs(repairs)


def repair_proven_income_candidate_versions():
    from core.data.income_data_migration import ensure_income_data_version_fields
    from core.domain.data_quality.income_versions import classify_income_version_candidates
    ensure_income_data_version_fields()
    candidate_path = "output/data_audit_live_20261025_candidates.json"
    shadow_path = "output/income_candidate_versions_shadow_20261025.json"
    candidate_bytes = load_bytes(candidate_path)
    shadow_bytes = load_bytes(shadow_path)
    candidate_data = load_json(candidate_path)
    shadow = load_json(shadow_path)
    if not candidate_bytes or not shadow_bytes or not candidate_data or not shadow:
        raise RuntimeError("候选清单或版本影子文件缺失")
    candidate_sha = __import__("hashlib").sha256(candidate_bytes).hexdigest()
    shadow_sha = __import__("hashlib").sha256(shadow_bytes).hexdigest()
    if shadow.get("candidate_sha256") != candidate_sha:
        raise RuntimeError("来源影子版本与当前候选清单不匹配")
    if shadow.get("codes_failed"):
        raise RuntimeError(f"TuShare 来源仍有失败或截断证券: {shadow['codes_failed']}")
    candidate_rows = candidate_data.get("margin_mismatch_candidates") or []
    candidate_keys = {(str(row["code"]), str(row["report_date"])[:10]) for row in candidate_rows}
    codes = shadow.get("codes") or {}
    source_by_key = {}
    source_count = 0
    for code, state in codes.items():
        if state.get("status") != "complete":
            raise RuntimeError(f"TuShare 来源状态未完成: {code}/{state.get('status')}")
        for record in state.get("records", []):
            period = str(record.get("end_date") or "").replace("-", "")[:8]
            if len(period) != 8:
                continue
            key = (str(record.get("ts_code") or "").split(".")[0], f"{period[:4]}-{period[4:6]}-{period[6:8]}")
            source_by_key.setdefault(key, []).append(record)
            source_count += 1
    if not candidate_keys.issubset(source_by_key):
        missing = sorted(candidate_keys - set(source_by_key))
        raise RuntimeError(f"候选来源覆盖不完整: {len(missing)} 条")
    current_rows = get_income_version_repair_rows(sorted(candidate_keys))
    current_by_key = {(row["code"], str(row["end_date"])): row for row in current_rows}
    if set(current_by_key) != candidate_keys:
        raise RuntimeError(f"正式利润表候选行覆盖不完整: {len(candidate_keys - set(current_by_key))} 条")
    actions, ledger = classify_income_version_candidates(candidate_rows, source_by_key, current_by_key)
    counts = {}
    for row in ledger:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    ledger_path = "output/income_candidate_version_adjudication_20261025.json"
    backup_path = "output/income_candidate_version_prewrite_20261025.json"
    action_keys = {(item["code"], item["end_date"]) for item in actions}
    ledger_artifact = {
        "candidate_sha256": candidate_sha,
        "source_sha256": shadow_sha,
        "candidate_keys": len(candidate_keys),
        "source_rows": source_count,
        "classification_counts": counts,
        "rows": ledger,
    }
    save_json_atomic(json.loads(json.dumps(ledger_artifact, ensure_ascii=False, default=str)), ledger_path)
    backup = {
        "candidate_sha256": candidate_sha,
        "source_sha256": shadow_sha,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": [
            {"code": item["code"], "end_date": item["end_date"], "status": item["status"],
             "before": item["expected"], "selected": item["selected"]}
            for item in actions
        ],
    }
    backup = json.loads(json.dumps(backup, ensure_ascii=False, default=str))
    existing_backup = load_json(backup_path)
    if existing_backup:
        if (existing_backup.get("candidate_sha256") != candidate_sha
                or existing_backup.get("source_sha256") != shadow_sha):
            raise RuntimeError("已存在的候选修复备份与本批来源不符")
        prior = {(row["code"], row["end_date"]): row for row in existing_backup.get("rows", [])}
        fields = ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost",
                  "operate_profit", "sell_exp", "admin_exp", "fin_exp")
        numeric = {"total_revenue", "oper_cost", "operate_profit", "sell_exp", "admin_exp", "fin_exp"}
        def _signature(row):
            return tuple(
                str(Decimal(str(row.get(field))).normalize()) if field in numeric and row.get(field) is not None
                else str(row.get(field).isoformat()) if isinstance(row.get(field), datetime.date)
                else str(row.get(field) or "")
                for field in fields
            )
        for key, old in prior.items():
            current = current_by_key.get(key)
            if not current:
                raise RuntimeError(f"备份中的正式行已不存在: {key[0]}/{key[1]}")
            if _signature(current) not in (_signature(old["before"]), _signature(old["selected"])):
                raise RuntimeError(f"正式数据偏离备份与选定版本，停止重跑: {key[0]}/{key[1]}")
    else:
        save_json_atomic(backup, backup_path)
    applied = apply_income_version_repairs(actions) if actions else []
    final_rows = get_income_version_repair_rows(sorted(candidate_keys))
    final_by_key = {(row["code"], str(row["end_date"])): row for row in final_rows}
    final_actions, final_ledger = classify_income_version_candidates(candidate_rows, source_by_key, final_by_key)
    if final_actions:
        raise RuntimeError(f"修复后仍有可按来源对齐的记录: {len(final_actions)} 条")
    final_counts = {}
    for row in final_ledger:
        final_counts[row["status"]] = final_counts.get(row["status"], 0) + 1
    result = {
        "candidate_keys": len(candidate_keys),
        "source_rows": source_count,
        "classification_before": counts,
        "rows_written": len(applied),
        "rows_verified": len(actions),
        "classification_after": final_counts,
        "unresolved": [row for row in final_ledger if row["status"] not in ("already_current",)],
        "backup_path": backup_path,
        "ledger_path": ledger_path,
    }
    result_path = "output/income_candidate_version_repair_result_20261025.json"
    save_json_atomic(json.loads(json.dumps(result, ensure_ascii=False, default=str)), result_path)
    result["result_path"] = result_path
    result["actions_by_status"] = {key: value for key, value in counts.items() if key != "already_current"}
    return result


def repair_officially_resolved_income_conflict():
    source_path = "output/income_candidate_versions_shadow_20261025.json"
    source_bytes = load_bytes(source_path)
    shadow = load_json(source_path)
    if not source_bytes or not shadow:
        raise RuntimeError("利润表来源影子文件不存在")
    source_records = (shadow.get("codes", {}).get("001872", {})
                      .get("records", []))
    official = {
        "code": "001872",
        "end_date": "2026-06-30",
        "f_ann_date": "20260829",
        "total_revenue": Decimal("8948462047.19"),
        "oper_cost": Decimal("4752291707.16"),
    }
    matches = [row for row in source_records
               if str(row.get("end_date")) == "20260630"
               and str(row.get("f_ann_date")) == official["f_ann_date"]
               and str(row.get("report_type")) == "1"
               and str(row.get("comp_type")) == "1"
               and Decimal(str(row.get("total_revenue"))) == official["total_revenue"]
               and Decimal(str(row.get("oper_cost"))) == official["oper_cost"]]
    if len(matches) != 1:
        raise RuntimeError("官方半年报金额未能唯一匹配到利润表源版本")
    selected_source = matches[0]
    current_rows = get_income_version_repair_rows([(official["code"], official["end_date"])])
    if len(current_rows) != 1:
        raise RuntimeError("正式利润表行缺失或不唯一")
    current = current_rows[0]
    selected = {
        "ann_date": datetime.datetime.strptime(str(selected_source["ann_date"]), "%Y%m%d").date(),
        "f_ann_date": datetime.datetime.strptime(str(selected_source["f_ann_date"]), "%Y%m%d").date(),
        "update_flag": str(selected_source.get("update_flag") or ""),
        **{field: selected_source.get(field) for field in (
            "total_revenue", "oper_cost", "operate_profit", "sell_exp", "admin_exp", "fin_exp")},
    }
    backup_path = "output/income_001872_20260630_prewrite_20261025.json"
    backup = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_path": source_path,
        "source_sha256": __import__("hashlib").sha256(source_bytes).hexdigest(),
        "official_report_url": "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12573054&stockid=001872",
        "official_revenue": str(official["total_revenue"]),
        "official_operating_cost": str(official["oper_cost"]),
        "before": current,
        "selected": selected,
        "selected_source": selected_source,
        "all_source_variants": source_records,
    }
    existing_backup = load_json(backup_path)
    if existing_backup:
        if existing_backup.get("source_sha256") != backup["source_sha256"]:
            raise RuntimeError("已有 001872 修复备份对应不同来源文件")
        before = existing_backup.get("before", {})
        selected_before = existing_backup.get("selected", {})
        current_values = (str(current.get("f_ann_date")), str(current.get("total_revenue")),
                          str(current.get("oper_cost")))
        allowed_values = {
            (str(before.get("f_ann_date")), str(before.get("total_revenue")), str(before.get("oper_cost"))),
            (str(selected_before.get("f_ann_date")), str(selected_before.get("total_revenue")),
             str(selected_before.get("oper_cost"))),
        }
        if current_values not in allowed_values:
            raise RuntimeError("正式行已偏离原值及已选来源，停止修复")
    else:
        save_json_atomic(json.loads(json.dumps(backup, ensure_ascii=False, default=str)), backup_path)
    action = {
        "code": official["code"],
        "end_date": official["end_date"],
        "expected": current,
        "selected": selected,
        "source_rows": source_records,
    }
    return apply_income_version_repairs([action])


def repair_officially_restated_income_rows(evidence_path="output/income_official_restatement_300076_20261025.json"):
    from core.data.income_data_migration import ensure_income_data_version_fields

    ensure_income_data_version_fields()
    evidence_bytes = load_bytes(evidence_path)
    evidence = load_json(evidence_path)
    if not evidence_bytes or not evidence or not evidence.get("rows"):
        raise RuntimeError("官方更正报告证据文件不存在或为空")
    rows = evidence["rows"]
    keys = [(item["code"], item["end_date"]) for item in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("官方更正报告包含重复证券/报告期")
    current_rows = get_income_version_repair_rows(keys)
    current_by_key = {(row["code"], str(row["end_date"])): row for row in current_rows}
    if set(current_by_key) != set(keys):
        raise RuntimeError("官方更正目标正式行缺失或重复")

    fields = ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost",
              "operate_profit", "sell_exp", "admin_exp", "fin_exp")
    numeric = set(fields) - {"ann_date", "f_ann_date", "update_flag"}
    actions = []
    selected_by_key = {}
    for item in rows:
        key = (item["code"], item["end_date"])
        current = current_by_key[key]
        source = item["source_record"]
        selected = item["selected"]
        if (source.get("ts_code", "").split(".")[0] != item["code"]
                or str(source.get("end_date")) != item["end_date"].replace("-", "")
                or str(source.get("f_ann_date")) != item["f_ann_date"].replace("-", "")
                or str(source.get("report_type")) != "1"
                or str(source.get("comp_type")) != "1"):
            raise RuntimeError(f"官方来源版本元数据不匹配: {key[0]}/{key[1]}")
        for field in numeric:
            if Decimal(str(source.get(field))) != Decimal(str(selected[field])):
                raise RuntimeError(f"官方来源记录与拟修复字段不匹配: {key[0]}/{key[1]}/{field}")
        if str(current.get("ann_date")) != item["ann_date"]:
            raise RuntimeError(f"正式公告日期与原报告不匹配: {key[0]}/{key[1]}")
        normalized = {
            "ann_date": datetime.datetime.strptime(item["ann_date"], "%Y-%m-%d").date(),
            "f_ann_date": datetime.datetime.strptime(item["f_ann_date"], "%Y-%m-%d").date(),
            "update_flag": str(selected["update_flag"]),
            **{field: Decimal(str(selected[field])) for field in numeric},
        }
        selected_by_key[key] = normalized
        actions.append({"code": item["code"], "end_date": item["end_date"],
                        "expected": current, "selected": normalized, "source_rows": [source]})

    backup_path = evidence.get("backup_path") or evidence_path.replace(".json", "_prewrite.json")
    backup = {
        "evidence_sha256": __import__("hashlib").sha256(evidence_bytes).hexdigest(),
        "official_report_url": evidence["official_report_url"],
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": [
            {"code": item["code"], "end_date": item["end_date"],
             "before": {field: str(current_by_key[(item["code"], item["end_date"])].get(field))
                        if current_by_key[(item["code"], item["end_date"])].get(field) is not None else None
                        for field in fields},
             "selected": {field: str(selected_by_key[(item["code"], item["end_date"])].get(field))
                          if selected_by_key[(item["code"], item["end_date"])].get(field) is not None else None
                          for field in fields}}
            for item in rows
        ],
    }
    existing = load_json(backup_path)
    if existing:
        if existing.get("evidence_sha256") != backup["evidence_sha256"]:
            raise RuntimeError("已有官方更正备份对应不同证据文件")
        prior = {(item["code"], item["end_date"]): item for item in existing.get("rows", [])}
        for item in backup["rows"]:
            saved = prior.get((item["code"], item["end_date"]))
            if not saved:
                raise RuntimeError(f"正式行已偏离备份和更正值: {item['code']}/{item['end_date']}")
            allowed = (saved.get("before") or {}, saved.get("selected") or {})
            matches = any(all(
                (value is None and candidate.get(field) is None)
                or (Decimal(str(value)) == Decimal(str(candidate.get(field))))
                if field in numeric and value is not None and candidate.get(field) is not None
                else value == candidate.get(field)
                for field, value in item["before"].items()
            ) for candidate in allowed)
            if not matches:
                raise RuntimeError(f"正式行已偏离备份和更正值: {item['code']}/{item['end_date']}")
    else:
        save_json_atomic(json.loads(json.dumps(backup, ensure_ascii=False, default=str)), backup_path)

    pending = []
    for action in actions:
        current = current_by_key[(action["code"], action["end_date"])]
        selected = action["selected"]
        if any((Decimal(str(current.get(field))) != selected[field]) if field in numeric
               else current.get(field) != selected[field] for field in fields):
            pending.append(action)
    applied = apply_income_version_repairs(pending) if pending else []
    final_rows = get_income_version_repair_rows(keys)
    final_by_key = {(row["code"], str(row["end_date"])): row for row in final_rows}
    for key, expected in selected_by_key.items():
        actual = final_by_key[key]
        for field in fields:
            if field in numeric:
                same = actual.get(field) is not None and Decimal(str(actual[field])) == expected[field]
            else:
                same = actual.get(field) == expected[field]
            if not same:
                raise RuntimeError(f"正式利润表读回不匹配: {key[0]}/{key[1]}/{field}")
    return {"rows_verified": len(keys), "rows_written": len(applied), "applied": applied,
            "backup_path": backup_path, "evidence_path": evidence_path}


def repair_officially_blank_income_costs_688197():
    from core.data.connection import Database

    evidence_path = "output/income_official_blank_oper_cost_688197_20261026.json"
    backup_path = "output/income_official_blank_oper_cost_688197_20261026_prewrite.json"
    evidence_rows = [
        {"code": "688197", "end_date": "2023-09-30", "ann_date": "2023-10-31",
         "total_revenue": "5000000.00", "source_url": "https://static.cninfo.com.cn/finalpage/2024-10-28/1221531256.PDF",
         "evidence_text": "2024年第三季度合并利润表列示2023年前三季度营业收入5,000,000.00元，营业成本栏空白。"},
        {"code": "688197", "end_date": "2024-06-30", "ann_date": "2024-08-26",
         "total_revenue": "3800000.00", "source_url": "https://static.cninfo.com.cn/finalpage/2025-08-28/1224591394.PDF",
         "evidence_text": "2025年半年度报告合并利润表比较列列示2024年半年度营业收入3,800,000.00元，营业成本栏空白。"},
        {"code": "688197", "end_date": "2024-09-30", "ann_date": "2024-10-28",
         "total_revenue": "3800000.00", "source_url": "https://static.cninfo.com.cn/finalpage/2024-10-28/1221531256.PDF",
         "evidence_text": "2024年第三季度合并利润表列示2024年前三季度营业收入3,800,000.00元，营业成本栏空白。"},
        {"code": "688197", "end_date": "2025-03-31", "ann_date": "2025-04-28",
         "total_revenue": "2000000.00", "source_url": "https://static.cninfo.com.cn/finalpage/2025-04-28/1223347612.PDF",
         "evidence_text": "2025年第一季度合并利润表列示营业收入2,000,000.00元，营业成本栏空白。"},
        {"code": "688197", "end_date": "2025-06-30", "ann_date": "2025-08-28",
         "total_revenue": "2000000.00", "source_url": "https://static.cninfo.com.cn/finalpage/2025-08-28/1224591394.PDF",
         "evidence_text": "2025年半年度合并利润表列示营业收入2,000,000.00元，营业成本栏空白。"},
        {"code": "688197", "end_date": "2025-09-30", "ann_date": "2025-10-31",
         "total_revenue": "2000000.00", "source_url": "https://static.cninfo.com.cn/finalpage/2025-10-31/1224769923.PDF",
         "evidence_text": "2025年第三季度合并利润表列示前三季度营业收入2,000,000.00元，营业成本栏空白。"},
    ]
    keys = [(row["code"], row["end_date"]) for row in evidence_rows]
    current_rows = get_income_version_repair_rows(keys)
    current_by_key = {(row["code"], str(row["end_date"])): row for row in current_rows}
    if set(current_by_key) != set(keys):
        raise RuntimeError("正式利润表目标行不完整")
    margin_rows = Database.fetchall(
        "SELECT code,report_date,grossprofit_margin FROM fundamental_data "
        + "WHERE (code,report_date) IN (" + ",".join(["(%s,%s)"] * len(keys)) + ")",
        [value for key in keys for value in key],
    )
    margin_by_key = {(row["code"], str(row["report_date"])): row.get("grossprofit_margin") for row in margin_rows}
    if set(margin_by_key) != set(keys) or any(value is not None for value in margin_by_key.values()):
        raise RuntimeError("毛利率目标行缺失或已非空，停止空成本修复")
    repairs = []
    for evidence in evidence_rows:
        key = (evidence["code"], evidence["end_date"])
        current = current_by_key[key]
        if str(current["ann_date"]) != evidence["ann_date"]:
            raise RuntimeError(f"公告日与正式库不匹配: {key[0]}/{key[1]}")
        if Decimal(str(current["total_revenue"])) != Decimal(evidence["total_revenue"]):
            raise RuntimeError(f"营业收入与报告不匹配: {key[0]}/{key[1]}")
        if current.get("oper_cost") not in (None, Decimal("0"), 0):
            raise RuntimeError(f"正式成本既非0也非空: {key[0]}/{key[1]}")
        source_row = {
            "ts_code": "688197.SH", "end_date": evidence["end_date"],
            "ann_date": evidence["ann_date"], "report_type": "1", "comp_type": "1",
            "source_name": "cninfo_official_report_blank_oper_cost",
            "source_url": evidence["source_url"], "evidence_text": evidence["evidence_text"],
            "evidence_observed_at": "2026-10-26", "oper_cost": None,
        }
        repairs.append({
            "code": key[0], "end_date": key[1], "expected": current,
            "selected": {"oper_cost": None}, "source_rows": [source_row],
        })
    evidence = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "688197 official consolidated statements with blank operating cost; cost remains missing and GPM is not imputed",
        "rows": evidence_rows,
    }
    evidence_bytes = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence_sha256 = __import__("hashlib").sha256(
        json.dumps(evidence_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    save_json_atomic(json.loads(evidence_bytes.decode("utf-8")), evidence_path)
    backup = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_path": evidence_path, "evidence_sha256": evidence_sha256,
        "rows": [
            {"code": item["code"], "end_date": str(item["end_date"]),
             "before": {field: str(item["expected"].get(field)) if item["expected"].get(field) is not None else None
                        for field in ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost")},
             "selected": {"oper_cost": None}}
            for item in repairs
        ],
    }
    existing_backup = load_json(backup_path)
    if existing_backup:
        if existing_backup.get("evidence_sha256") != evidence_sha256:
            raise RuntimeError("既有写前备份对应不同证据，停止重跑")
        prior = {(row["code"], row["end_date"]): row for row in existing_backup.get("rows", [])}
        for item in repairs:
            saved = prior.get((item["code"], item["end_date"]))
            if not saved:
                raise RuntimeError(f"写前备份缺少目标行: {item['code']}/{item['end_date']}")
            before_cost = saved.get("before", {}).get("oper_cost")
            current_cost = item["expected"].get("oper_cost")
            if current_cost is not None and Decimal(str(current_cost)) != Decimal(str(before_cost)):
                raise RuntimeError(f"正式成本偏离备份: {item['code']}/{item['end_date']}")
    else:
        save_json_atomic(json.loads(json.dumps(backup, ensure_ascii=False, default=str)), backup_path)
    pending = [item for item in repairs if item["expected"].get("oper_cost") is not None]
    applied = apply_income_version_repairs(pending) if pending else []
    final_rows = get_income_version_repair_rows(keys)
    final_by_key = {(row["code"], str(row["end_date"])): row for row in final_rows}
    if any(final_by_key[key].get("oper_cost") is not None for key in keys):
        raise RuntimeError("正式利润表成本空值读回不一致")
    return {"rows_verified": len(keys), "rows_written": len(applied), "applied": applied,
            "backup_path": backup_path, "evidence_path": evidence_path, "evidence_sha256": evidence_sha256}


def repair_officially_blank_income_cost_688197_2022q3():
    evidence_path = "output/income_official_blank_oper_cost_688197_2022q3_20261026.json"
    backup_path = "output/income_official_blank_oper_cost_688197_2022q3_20261026_prewrite.json"
    key = ("688197", "2022-09-30")
    current_rows = get_income_version_repair_rows([key])
    if len(current_rows) != 1:
        raise RuntimeError("正式利润表目标行缺失或重复")
    current = current_rows[0]
    if (str(current.get("ann_date")) != "2022-10-31"
            or Decimal(str(current.get("total_revenue"))) != Decimal("1785218.98")
            or current.get("oper_cost") not in (Decimal("0"), 0, None)):
        raise RuntimeError("正式行与2022Q3报告不一致，停止修复")
    source_row = {
        "ts_code": "688197.SH", "end_date": "2022-09-30", "ann_date": "2022-10-31",
        "report_type": "1", "comp_type": "1", "source_name": "cninfo_official_report_blank_oper_cost",
        "source_url": "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=8640156&stockid=688197",
        "evidence_text": "2022年第三季度合并利润表列示前三季度营业收入1,785,218.98元，营业成本栏空白；本次仅将该期供应商零值改为NULL。",
        "evidence_observed_at": "2026-10-26", "oper_cost": None,
    }
    evidence = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "official 2022Q3 consolidated income statement blank operating cost",
        "rows": [{"code": key[0], "end_date": key[1], "ann_date": "2022-10-31",
                  "total_revenue": "1785218.98", "source_url": source_row["source_url"],
                  "evidence_text": source_row["evidence_text"]}],
    }
    evidence_rows_bytes = json.dumps(evidence["rows"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence_sha256 = __import__("hashlib").sha256(evidence_rows_bytes).hexdigest()
    save_json_atomic(evidence, evidence_path)
    backup = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_path": evidence_path, "evidence_sha256": evidence_sha256,
        "rows": [{"code": key[0], "end_date": key[1],
                  "before": {field: str(current.get(field)) if current.get(field) is not None else None
                             for field in ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost")},
                  "selected": {"oper_cost": None}}],
    }
    existing_backup = load_json(backup_path)
    if existing_backup:
        if existing_backup.get("evidence_sha256") != evidence_sha256:
            raise RuntimeError("既有写前备份对应不同证据，停止重跑")
        saved = existing_backup.get("rows", [{}])[0].get("before", {}).get("oper_cost")
        value = current.get("oper_cost")
        if value is not None and Decimal(str(value)) != Decimal(str(saved)):
            raise RuntimeError("正式成本偏离写前备份，停止重跑")
    else:
        save_json_atomic(json.loads(json.dumps(backup, ensure_ascii=False, default=str)), backup_path)
    applied = apply_income_version_repairs([{
        "code": key[0], "end_date": key[1], "expected": current,
        "selected": {"oper_cost": None}, "source_rows": [source_row],
    }]) if current.get("oper_cost") is not None else []
    final = get_income_version_repair_rows([key])
    if len(final) != 1 or final[0].get("oper_cost") is not None:
        raise RuntimeError("正式库营业成本空值读回不一致")
    return {"rows_verified": 1, "rows_written": len(applied), "applied": applied,
            "backup_path": backup_path, "evidence_path": evidence_path, "evidence_sha256": evidence_sha256}


def repair_officially_restated_income_fields(evidence_path):
    from core.data.income_data_migration import ensure_income_data_version_fields

    ensure_income_data_version_fields()
    evidence_bytes = load_bytes(evidence_path)
    evidence = load_json(evidence_path)
    if not evidence_bytes or not evidence or not evidence.get("rows"):
        raise RuntimeError("官方更正报告证据文件不存在或为空")
    allowed_numeric = {"total_revenue", "oper_cost", "operate_profit", "sell_exp", "admin_exp", "fin_exp"}
    rows = evidence["rows"]
    keys = [(item["code"], item["end_date"]) for item in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("官方更正报告包含重复证券/报告期")
    current_rows = get_income_version_repair_rows(keys)
    current_by_key = {(row["code"], str(row["end_date"])): row for row in current_rows}
    if set(current_by_key) != set(keys):
        raise RuntimeError("官方更正目标正式行缺失或重复")

    fields = ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost",
              "operate_profit", "sell_exp", "admin_exp", "fin_exp")
    actions = []
    selected_by_key = {}
    for item in rows:
        key = (item["code"], item["end_date"])
        current = current_by_key[key]
        source = item["source_record"]
        selected = item["selected"]
        numeric_fields = set(item.get("fields") or [])
        if not numeric_fields or not numeric_fields.issubset(allowed_numeric):
            raise RuntimeError(f"官方更正字段清单为空或含非法字段: {key[0]}/{key[1]}")
        if (source.get("ts_code", "").split(".")[0] != item["code"]
                or str(source.get("ann_date")) != item["ann_date"].replace("-", "")
                or str(source.get("end_date")) != item["end_date"].replace("-", "")
                or str(source.get("f_ann_date")) != item["f_ann_date"].replace("-", "")
                or str(source.get("report_type")) != "1"
                or str(source.get("comp_type")) != "1"):
            raise RuntimeError(f"官方来源版本元数据不匹配: {key[0]}/{key[1]}")
        if str(current.get("ann_date")) != item["ann_date"]:
            raise RuntimeError(f"正式公告日期与原报告不匹配: {key[0]}/{key[1]}")
        if (str(selected.get("f_ann_date")) != item["f_ann_date"].replace("-", "")
                or str(source.get("update_flag")) != str(selected.get("update_flag"))):
            raise RuntimeError(f"官方来源版本标记不匹配: {key[0]}/{key[1]}")
        for field in numeric_fields:
            if field not in selected or source.get(field) is None or Decimal(str(source[field])) != Decimal(str(selected[field])):
                raise RuntimeError(f"官方来源记录与拟修复字段不匹配: {key[0]}/{key[1]}/{field}")
        normalized = {
            "ann_date": datetime.datetime.strptime(item["ann_date"], "%Y-%m-%d").date(),
            "f_ann_date": datetime.datetime.strptime(item["f_ann_date"], "%Y-%m-%d").date(),
            "update_flag": str(selected["update_flag"]),
            **{field: Decimal(str(selected[field])) for field in numeric_fields},
        }
        desired = {field: current.get(field) for field in fields}
        desired.update(normalized)
        selected_by_key[key] = desired
        actions.append({"code": item["code"], "end_date": item["end_date"],
                        "expected": current, "selected": normalized, "source_rows": [source]})

    backup_path = evidence.get("backup_path") or evidence_path.replace(".json", "_prewrite.json")
    backup = {
        "evidence_sha256": __import__("hashlib").sha256(evidence_bytes).hexdigest(),
        "official_report_url": evidence["official_report_url"],
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": [
            {"code": item["code"], "end_date": item["end_date"],
             "before": {field: str(current_by_key[(item["code"], item["end_date"])].get(field))
                        if current_by_key[(item["code"], item["end_date"])].get(field) is not None else None
                        for field in fields},
             "selected": {field: str(selected_by_key[(item["code"], item["end_date"])].get(field))
                          if selected_by_key[(item["code"], item["end_date"])].get(field) is not None else None
                          for field in fields}}
            for item in rows
        ],
    }
    existing = load_json(backup_path)
    if existing:
        if existing.get("evidence_sha256") != backup["evidence_sha256"]:
            raise RuntimeError("已有官方更正备份对应不同证据文件")
        prior = {(item["code"], item["end_date"]): item for item in existing.get("rows", [])}
        for item in backup["rows"]:
            saved = prior.get((item["code"], item["end_date"]))
            if not saved:
                raise RuntimeError(f"备份中缺少目标行: {item['code']}/{item['end_date']}")
            matches = False
            for candidate in (saved.get("before") or {}, saved.get("selected") or {}):
                if all(
                        Decimal(str(item["before"][field])) == Decimal(str(candidate[field]))
                        if field in allowed_numeric and item["before"][field] is not None and candidate.get(field) is not None
                        else item["before"][field] == candidate.get(field)
                        for field in fields):
                    matches = True
                    break
            if not matches:
                raise RuntimeError(f"正式行已偏离备份和更正值: {item['code']}/{item['end_date']}")
    else:
        save_json_atomic(json.loads(json.dumps(backup, ensure_ascii=False, default=str)), backup_path)

    pending = []
    for action in actions:
        current = current_by_key[(action["code"], action["end_date"])]
        if any(current.get(field) != value for field, value in action["selected"].items()):
            pending.append(action)
    applied = apply_income_version_repairs(pending) if pending else []
    final_rows = get_income_version_repair_rows(keys)
    final_by_key = {(row["code"], str(row["end_date"])): row for row in final_rows}
    for key, expected in selected_by_key.items():
        for field in fields:
            value = expected[field]
            actual = final_by_key[key].get(field)
            if field in allowed_numeric and value is not None:
                same = actual is not None and Decimal(str(actual)) == Decimal(str(value))
            else:
                same = actual == value
            if not same:
                raise RuntimeError(f"正式利润表读回不匹配: {key[0]}/{key[1]}/{field}")
    return {"rows_verified": len(keys), "rows_written": len(applied), "applied": applied,
            "backup_path": backup_path, "evidence_path": evidence_path}


def repair_officially_verified_income_fields(evidence_path):
    from core.data.income_data_migration import ensure_income_data_version_fields

    ensure_income_data_version_fields()
    evidence_bytes = load_bytes(evidence_path)
    evidence = load_json(evidence_path)
    rows = evidence.get("rows") if evidence else None
    if not evidence_bytes or not evidence or not evidence.get("official_report_url") or not rows:
        raise RuntimeError("官方利润表字段证据文件不存在或不完整")
    allowed = {"total_revenue", "oper_cost", "operate_profit", "sell_exp", "admin_exp", "fin_exp"}
    keys = [(item["code"], item["end_date"]) for item in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("官方利润表证据包含重复证券/报告期")
    current_rows = get_income_version_repair_rows(keys)
    current_by_key = {(row["code"], str(row["end_date"])): row for row in current_rows}
    if set(current_by_key) != set(keys):
        raise RuntimeError("官方利润表目标正式行缺失或重复")

    def same(left, right):
        if left is None or right is None:
            return left is right
        try:
            return Decimal(str(left)) == Decimal(str(right))
        except Exception:
            return str(left)[:10] == str(right)[:10]

    actions = []
    backup_rows = []
    for item in rows:
        key = (item["code"], item["end_date"])
        current = current_by_key[key]
        if str(current.get("ann_date")) != item["ann_date"]:
            raise RuntimeError(f"正式利润表公告日与官方证据不匹配: {key[0]}/{key[1]}")
        selected = item.get("selected") or {}
        source = item.get("source_record") or {}
        if not selected or not set(selected).issubset(allowed):
            raise RuntimeError(f"官方利润表选值字段为空或非法: {key[0]}/{key[1]}")
        if (source.get("ts_code", "").split(".")[0] != item["code"]
                or str(source.get("end_date")) != item["end_date"].replace("-", "")
                or str(source.get("ann_date")) != item["ann_date"].replace("-", "")
                or source.get("report_type") != "1"
                or source.get("comp_type") != "1"
                or source.get("source_url") != evidence["official_report_url"]):
            raise RuntimeError(f"官方利润表来源记录元数据不匹配: {key[0]}/{key[1]}")
        for field, value in selected.items():
            if value is None or source.get(field) is None or not same(source[field], value):
                raise RuntimeError(f"官方来源记录与选值不匹配: {key[0]}/{key[1]}/{field}")
            expected_value = (item.get("expected") or {}).get(field)
            if (field in (item.get("expected") or {})
                    and not same(current.get(field), expected_value)
                    and not same(current.get(field), value)):
                raise RuntimeError(f"正式利润表行偏离写前值: {key[0]}/{key[1]}/{field}")
        expected = dict(current)
        backup_rows.append({
            "code": item["code"], "end_date": item["end_date"],
            "before": {field: str(current.get(field)) if current.get(field) is not None else None
                       for field in ("ann_date", "f_ann_date", "update_flag", "total_revenue", "oper_cost",
                                     "operate_profit", "sell_exp", "admin_exp", "fin_exp")},
            "selected": {field: str(value) if value is not None else None for field, value in selected.items()},
        })
        actions.append({"code": item["code"], "end_date": item["end_date"], "expected": expected,
                        "selected": {field: Decimal(str(value)) if value is not None else None
                                     for field, value in selected.items()},
                        "source_rows": [source]})

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
            raise RuntimeError("已有官方利润表写前备份对应不同证据")
        prior = {(row["code"], row["end_date"]): row for row in existing.get("rows", [])}
        for row in backup_rows:
            old = prior.get((row["code"], row["end_date"]))
            if not old:
                raise RuntimeError(f"正式利润表行偏离写前备份: {row['code']}/{row['end_date']}")
            old_before = old.get("before") or {}
            old_selected = old.get("selected") or {}
            for field, value in row["before"].items():
                if not same(value, old_before.get(field)) and (
                        field not in old_selected or not same(value, old_selected[field])):
                    raise RuntimeError(f"正式利润表行偏离写前备份: {row['code']}/{row['end_date']}/{field}")
    else:
        save_json_atomic(backup, backup_path)

    applied = apply_income_version_repairs(actions) if actions else []
    final_rows = get_income_version_repair_rows(keys)
    final_by_key = {(row["code"], str(row["end_date"])): row for row in final_rows}
    for action in actions:
        for field, value in action["selected"].items():
            if not same(final_by_key[(action["code"], action["end_date"])].get(field), value):
                raise RuntimeError(f"正式利润表官方选值读回不匹配: {action['code']}/{action['end_date']}/{field}")
    return {"rows_verified": len(keys),
            "rows_written": sum(row.get("status") == "repaired" for row in applied),
            "source_versions_registered": sum(row.get("variant_rows", 0) for row in applied),
            "applied": applied,
            "backup_path": backup_path, "evidence_path": evidence_path}
