import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from core.data.repos.historical_repair import HistoricalRepairRepo
from core.shared import config
from core.shared.persistence import load_json, load_parquet, now_iso, save_json_atomic
from backend.service import historical_repair as repair


def publish_historical_release_batch(release_ids):
    release_ids = list(dict.fromkeys(str(value).strip() for value in release_ids))
    if not release_ids or any(not value.isalnum() or len(value) != 12 for value in release_ids):
        raise ValueError("批量历史发布 ID 列表无效")
    with repair._release_publish_lock:
        with repair._historical_release_advisory_lock("historical_batch_release"):
            manifests = []
            codes = set()
            for release_id in release_ids:
                manifest = repair.get_historical_release(release_id)
                gate_status, gate_blockers = repair._historical_release_gate(
                    [manifest.get("blockers") or []],
                    (manifest.get("assessment") or {}).get("status"),
                    (manifest.get("qlib") or {}).get("status"),
                    [manifest.get("blocker_rule_ids") or []],
                )
                if (
                    manifest.get("release_id") != release_id
                    or manifest.get("status") != "ready"
                    or not manifest.get("publishable")
                    or manifest.get("release_change_contract") != "g2_effective_change_v1"
                    or gate_status != "ready"
                ):
                    detail = f"；阻塞项：{gate_blockers}" if gate_blockers else ""
                    raise RuntimeError(f"发布预检 {release_id} 已过期或未通过{detail}")
                if manifest.get("full_stock_reanchor") or manifest.get("repair_fields"):
                    raise RuntimeError("批量发布仅接受标准缺行修复预检")
                if manifest.get("code") in codes:
                    raise ValueError(f"批量发布包含重复证券 {manifest.get('code')}")
                codes.add(manifest.get("code"))
                candidate_path = Path(manifest["candidate_path"])
                preimage_path = Path(manifest["preimage_path"])
                if not candidate_path.is_file() or repair._file_sha256(candidate_path) != manifest.get("candidate_sha256"):
                    raise RuntimeError(f"{manifest.get('code')} 候选文件哈希不符")
                if not preimage_path.is_file() or repair._file_sha256(preimage_path) != manifest.get("preimage_sha256"):
                    raise RuntimeError(f"{manifest.get('code')} pre-image 哈希不符")
                shadow_path = repair._manifest_path(manifest["batch_id"])
                shadow = load_json(shadow_path)
                item = next((row for row in (shadow or {}).get("work_items", [])
                             if row.get("code") == manifest.get("code")
                             and int(row.get("year", 0)) == int(manifest.get("year", 0))), None)
                if not item or not repair._shadow_files_valid(shadow_path.parent, item):
                    raise RuntimeError(f"{manifest.get('code')} G1 双源证据哈希不符")
                if repair._file_sha256(Path(manifest["source_candidate_path"])) != manifest.get("source_candidate_sha256"):
                    raise RuntimeError(f"{manifest.get('code')} G1 候选哈希不符")
                if repair._file_sha256(shadow_path.parent / "raw" / manifest["code"] / f"{manifest['year']}_comparison.parquet") != manifest.get("comparison_sha256"):
                    raise RuntimeError(f"{manifest.get('code')} 双源比较哈希不符")
                current = HistoricalRepairRepo.get_shard_inputs(
                    manifest["code"], manifest["date_range"]["start"], manifest["date_range"]["end"],
                )
                if current["fingerprint"] != manifest.get("input_fingerprint"):
                    raise RuntimeError(f"{manifest.get('code')} 正式库基线变化，需重新预检")
                manifests.append(manifest)

            from core.adapters.qlib.generation import (
                clear_historical_release_active,
                discard_qlib_generation_stage,
                has_qlib_generation_pointer,
                mark_historical_release_active,
                publish_qlib_generation,
                qlib_generation_mutation_lock,
                resolve_qlib_data_dir,
                rollback_qlib_generation,
                stage_qlib_generation,
            )

            if not has_qlib_generation_pointer(config.QLIB_DATA_ROOT):
                raise RuntimeError("批量历史发布要求受管 Qlib generation")
            parent_dir = resolve_qlib_data_dir(config.QLIB_DATA_ROOT)
            parent_id = parent_dir.name
            if any(m.get("qlib_generation_id") != parent_id for m in manifests):
                raise RuntimeError("批量预检的 Qlib 基线不一致，需基于当前活动代重新预检")

            batch_release_id = uuid.uuid4().hex[:12]
            release_dir = repair._release_root() / batch_release_id
            release_dir.mkdir(parents=True, exist_ok=False)
            path = release_dir / "manifest.json"
            stage_id = uuid.uuid4().hex
            stage_dir = Path(config.QLIB_DATA_ROOT).resolve() / "generations" / f".staging-{stage_id}"
            batch_manifest = {
                "schema_version": 1,
                "release_id": batch_release_id,
                "release_type": "batch",
                "status": "staging",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "child_release_ids": release_ids,
                "codes": sorted(codes),
                "qlib_generation_parent_id": parent_id,
                "qlib_generation_stage_id": stage_id,
                "qlib_generation_stage_dir": str(stage_dir),
                "qlib_generation_stage_state": "copying",
                "official_database_writes": False,
                "qlib_writes": False,
                "publishable": False,
                "database_results": [],
                "qlib": {"status": "ready", "files": {}},
                "blockers": [],
            }
            save_json_atomic(batch_manifest, path)
            stage = None
            db_results = []
            db_preimages = {}
            qlib_published_id = None
            active_marked = False
            qlib_lock_acquired = False
            try:
                stage = stage_qlib_generation(
                    data_root=config.QLIB_DATA_ROOT,
                    generation_id=stage_id,
                    expected_parent_generation_id=parent_id,
                    lifecycle_lease=True,
                    owner={"type": "historical_release", "release_id": batch_release_id},
                )
                if Path(stage["stage_dir"]).resolve() != stage_dir.resolve():
                    raise RuntimeError("Qlib generation 暂存目录身份不一致")

                instrument_lines = (parent_dir / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines()
                instrument_by_code = {line.split("\t", 1)[0].lower(): index for index, line in enumerate(instrument_lines) if line}
                instrument_changed = False
                patch_targets = set()
                merged_files = {}
                for manifest in manifests:
                    code = manifest["code"]
                    qlib_code = str(manifest["qlib"].get("qlib_code") or "").lower()
                    for field, record in manifest.get("qlib", {}).get("files", {}).items():
                        source = Path(record["source"]).resolve()
                        relative = source.relative_to(parent_dir.resolve())
                        parent_source = parent_dir / relative
                        current_hash = repair._file_sha256(parent_source) if parent_source.is_file() else None
                        if current_hash != record.get("before_sha256"):
                            raise RuntimeError(f"{code} Qlib {field} 发布前基线哈希变化")
                        staged_patch = Path(record["staged"])
                        if not staged_patch.is_file() or repair._file_sha256(staged_patch) != record.get("after_sha256"):
                            raise RuntimeError(f"{code} Qlib {field} 补丁哈希不符")
                        if field == "instruments":
                            staged_lines = staged_patch.read_text(encoding="utf-8").splitlines()
                            matches = [line for line in staged_lines if line.split("\t", 1)[0].lower() == qlib_code]
                            if len(matches) != 1 or qlib_code not in instrument_by_code:
                                raise RuntimeError(f"{code} Qlib instruments 行无法唯一合并")
                            instrument_lines[instrument_by_code[qlib_code]] = matches[0]
                            instrument_changed = True
                            continue
                        target = stage_dir / relative
                        target_key = relative.as_posix().lower()
                        if target_key in patch_targets:
                            raise RuntimeError(f"多个预检修改同一 Qlib 文件: {relative}")
                        patch_targets.add(target_key)
                        staged_current = repair._file_sha256(target) if target.is_file() else None
                        if staged_current != record.get("before_sha256"):
                            raise RuntimeError(f"Qlib 候选代 {relative} 基线哈希不符")
                        shutil.copy2(staged_patch, target)
                        if repair._file_sha256(target) != record.get("after_sha256"):
                            raise RuntimeError(f"Qlib 候选代 {relative} 补丁读回哈希不符")
                        merged_files[f"{code}:{field}"] = {
                            "source": str(target),
                            "source_before_generation": str(parent_source),
                            "generation_relative_path": relative.as_posix(),
                            "before_sha256": record.get("before_sha256"),
                            "after_sha256": record.get("after_sha256"),
                        }
                if instrument_changed:
                    relative = Path("instruments/all.txt")
                    target = stage_dir / relative
                    parent_source = parent_dir / relative
                    before_hash = repair._file_sha256(parent_source)
                    target.write_text("\n".join(instrument_lines) + "\n", encoding="utf-8")
                    merged_files["instruments:all"] = {
                        "source": str(target),
                        "source_before_generation": str(parent_source),
                        "generation_relative_path": relative.as_posix(),
                        "before_sha256": before_hash,
                        "after_sha256": repair._file_sha256(target),
                    }
                batch_manifest["qlib"]["files"] = merged_files
                batch_manifest["qlib_generation_stage_state"] = "ready"
                batch_manifest["updated_at"] = now_iso()
                save_json_atomic(batch_manifest, path)

                qlib_lock = qlib_generation_mutation_lock(config.QLIB_DATA_ROOT)
                qlib_lock.__enter__()
                qlib_lock_acquired = True
                mark_historical_release_active(batch_release_id, path, data_root=config.QLIB_DATA_ROOT)
                active_marked = True
                batch_manifest["status"] = "publishing"
                batch_manifest["publish_started_at"] = now_iso()
                save_json_atomic(batch_manifest, path)

                for manifest in manifests:
                    code = manifest["code"]
                    candidate = load_parquet(manifest["candidate_path"])
                    before_rows = load_parquet(manifest["preimage_path"]).to_dict("records")
                    shadow = load_json(repair._manifest_path(manifest["batch_id"]))
                    provenance = {
                        "batch_id": manifest["release_id"],
                        "task_id": f"historical-{manifest['release_id']}",
                        "source": "tushare+baostock",
                        "provider_version": json.dumps(shadow["source_versions_at_collection"], sort_keys=True),
                        "adjustment_mode": "qfq",
                        "anchor_date": manifest["price_basis"]["anchor_date"],
                        "input_payload_sha256": manifest["candidate_sha256"],
                        "source_verified_conflict_replacement": False,
                    }
                    db_preimages[code] = {"manifest": manifest, "rows": before_rows, "provenance": provenance}
                    result = HistoricalRepairRepo.publish_missing_values(
                        code, manifest["date_range"]["start"], manifest["date_range"]["end"],
                        candidate, manifest["input_fingerprint"], provenance,
                    )
                    db_results.append({"code": code, "release_id": manifest["release_id"], "result": result, "provenance": provenance})
                    batch_manifest["database_results"] = [
                        {"code": row["code"], "release_id": row["release_id"], "input_rows": row["result"]["input_rows"],
                         "after_fingerprint": row["result"]["after_fingerprint"]}
                        for row in db_results
                    ]
                    batch_manifest["status"] = "database_publishing"
                    batch_manifest["updated_at"] = now_iso()
                    save_json_atomic(batch_manifest, path)

                batch_manifest["status"] = "database_published"
                save_json_atomic(batch_manifest, path)
                published = publish_qlib_generation(
                    stage,
                    metadata={"operation": "historical_quote_repair_batch", "release_id": batch_release_id,
                              "child_release_ids": release_ids},
                    expected_generation_id=parent_id,
                )
                qlib_published_id = published["generation_id"]
                batch_manifest["qlib_generation_published_id"] = qlib_published_id
                batch_manifest["qlib_dir"] = published["data_dir"]
                batch_manifest["status"] = "published"
                batch_manifest["official_database_writes"] = bool(db_results)
                batch_manifest["qlib_writes"] = bool(merged_files)
                batch_manifest["publishable"] = False
                batch_manifest["published_at"] = now_iso()
                batch_manifest["updated_at"] = batch_manifest["published_at"]
                save_json_atomic(batch_manifest, path)
                for manifest in manifests:
                    child_path = repair._release_manifest_path(manifest["release_id"])
                    db_result = next(row["result"] for row in db_results if row["release_id"] == manifest["release_id"])
                    for record in manifest.get("qlib", {}).get("files", {}).values():
                        relative = Path(record["source"]).resolve().relative_to(parent_dir.resolve())
                        record["source_before_generation"] = str((parent_dir / relative).resolve())
                        record["generation_relative_path"] = relative.as_posix()
                        record["source"] = str((Path(published["data_dir"]) / relative).resolve())
                    manifest["qlib_dir"] = published["data_dir"]
                    manifest["status"] = "published"
                    manifest["published_by_batch_release"] = batch_release_id
                    manifest["official_database_writes"] = True
                    manifest["qlib_writes"] = True
                    manifest["publishable"] = False
                    manifest["db_after_fingerprint"] = db_result["after_fingerprint"]
                    manifest["db_input_rows"] = db_result["input_rows"]
                    manifest["db_audit_rows"] = db_result["audit_rows"]
                    manifest["qlib_generation_published_id"] = qlib_published_id
                    manifest["published_at"] = batch_manifest["published_at"]
                    manifest["updated_at"] = batch_manifest["published_at"]
                    save_json_atomic(manifest, child_path)
                try:
                    clear_historical_release_active(batch_release_id, data_root=config.QLIB_DATA_ROOT)
                except Exception:
                    pass
                return batch_manifest
            except Exception as error:
                rollback_errors = []
                current_generation_id = (
                    resolve_qlib_data_dir(config.QLIB_DATA_ROOT).name
                    if has_qlib_generation_pointer(config.QLIB_DATA_ROOT) else None
                )
                if current_generation_id == stage_id:
                    try:
                        repair._verify_qlib_generation_release_files(batch_manifest, stage_id, "after")
                        repair._verify_qlib_generation_release_files(batch_manifest, parent_id, "before")
                        rollback_qlib_generation(parent_id, data_root=config.QLIB_DATA_ROOT,
                                                 expected_current_generation_id=stage_id)
                    except Exception as rollback_error:
                        rollback_errors.append(f"Qlib 回滚失败：{rollback_error}")
                elif current_generation_id != parent_id:
                    rollback_errors.append("Qlib 活动代与父代不一致，无法自动回滚")
                elif stage:
                    candidate_dir = Path(config.QLIB_DATA_ROOT).resolve() / "generations" / stage_id
                    if candidate_dir.is_dir():
                        try:
                            repair._discard_unpublished_qlib_generation_candidate(
                                batch_manifest, batch_release_id, stage_id,
                            )
                        except Exception as cleanup_error:
                            rollback_errors.append(f"Qlib 未激活候选代清理失败：{cleanup_error}")
                for row in reversed(db_results):
                    snapshot = db_preimages[row["code"]]
                    manifest = snapshot["manifest"]
                    try:
                        HistoricalRepairRepo.restore_shard(
                            row["code"], manifest["rollback_range"]["start"], manifest["rollback_range"]["end"],
                            snapshot["rows"], row["result"]["after_fingerprint"], row["provenance"],
                        )
                    except Exception as rollback_error:
                        rollback_errors.append(f"{row['code']} 数据库回滚失败：{rollback_error}")
                if stage and Path(stage["stage_dir"]).exists():
                    try:
                        discard_qlib_generation_stage(stage)
                    except Exception as cleanup_error:
                        rollback_errors.append(f"Qlib 暂存清理失败：{cleanup_error}")
                for release_id in release_ids:
                    child_path = repair._release_manifest_path(release_id)
                    child = repair.get_historical_release(release_id)
                    if child.get("published_by_batch_release") == batch_release_id:
                        child["status"] = "rollback_required" if rollback_errors else "ready"
                        child["publishable"] = not rollback_errors
                        child["official_database_writes"] = False
                        child["qlib_writes"] = False
                        child.pop("published_by_batch_release", None)
                        child.pop("qlib_generation_published_id", None)
                        child.pop("published_at", None)
                        child["updated_at"] = now_iso()
                        save_json_atomic(child, child_path)
                batch_manifest["status"] = "rollback_required" if rollback_errors else "rolled_back"
                batch_manifest["error"] = f"{type(error).__name__}: {error}"
                batch_manifest["rollback_errors"] = rollback_errors
                batch_manifest["updated_at"] = now_iso()
                save_json_atomic(batch_manifest, path)
                if active_marked and not rollback_errors:
                    try:
                        clear_historical_release_active(batch_release_id, data_root=config.QLIB_DATA_ROOT)
                    except Exception as cleanup_error:
                        batch_manifest["active_marker_cleanup_error"] = str(cleanup_error)
                        save_json_atomic(batch_manifest, path)
                raise RuntimeError(f"批量历史发布失败；状态 {batch_manifest['status']}；{batch_manifest['error']}；{rollback_errors}") from error
            finally:
                if qlib_lock_acquired:
                    qlib_lock.__exit__(None, None, None)
