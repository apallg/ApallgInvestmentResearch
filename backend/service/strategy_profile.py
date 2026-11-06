import datetime
import hashlib
import json
import time
import uuid

PIT_CONTRACT_VERSION = "source_available_at_and_reference_integrity_v3"


def _json_fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _present(row):
    if not row:
        return row
    result = dict(row)
    result["config"] = result.pop("config_json", {})
    result["blocked_reasons"] = result.pop("blocked_reasons_json", [])
    result["is_active"] = bool(result.get("is_active"))
    return result


def _security_reference_input_fingerprint(references):
    return _json_fingerprint(sorted(references, key=lambda row: row.get("code", "")))


def _security_reference_fingerprint(references):
    return _json_fingerprint([
        {
            "code": row.get("code"), "name": row.get("name", ""),
            "industry": row.get("industry", ""), "board": row.get("board", ""),
            "is_st": bool(row.get("is_st")), "is_suspended": bool(row.get("is_suspended")),
            "listed_date": row.get("listed_date"), "delisted_date": row.get("delisted_date"),
            "source": row.get("source"),
            "provenance_status": row.get("provenance_status", "unverified"),
            "source_name": row.get("source_name"), "source_version": row.get("source_version"),
            "source_available_at": row.get("source_available_at"),
            "source_evidence_sha256": row.get("source_evidence_sha256"),
        }
        for row in sorted(references, key=lambda item: item.get("code", ""))
    ])


class StrategyProfileService:
    _snapshot_integrity_cache = {}

    @staticmethod
    def snapshot_has_verified_pit(snapshot):
        if not snapshot or snapshot.get("pit_status") != "pass":
            return False
        if snapshot.get("pit_contract_version") != PIT_CONTRACT_VERSION:
            return False
        payload = snapshot.get("snapshot_json")
        if payload is None:
            payload = snapshot
        if not isinstance(payload, dict) or payload.get("pit_contract_version") != PIT_CONTRACT_VERSION:
            return False
        coverage = payload.get("security_reference_coverage")
        try:
            coverage = float(coverage)
        except (TypeError, ValueError):
            return False
        fingerprint = str(payload.get("security_reference_input_fingerprint") or "").lower()
        reference_fingerprint = str(payload.get("security_reference_fingerprint") or "").lower()
        return (
            payload.get("security_reference_status") == "pass"
            and 0.999 <= coverage <= 1.0
            and all(
                len(value) == 64 and all(char in "0123456789abcdef" for char in value)
                for value in (fingerprint, reference_fingerprint)
            )
        )

    @classmethod
    def snapshot_has_current_integrity(cls, snapshot):
        return cls.snapshot_has_verified_pit(snapshot) and cls._snapshot_content_matches(snapshot)

    @staticmethod
    def security_reference_has_verified_provenance(row, as_of):
        if (not row or row.get("source") != "pit_import"
                or row.get("provenance_status") != "verified"
                or not row.get("source_name") or not row.get("source_version")):
            return False
        evidence_hash = str(row.get("source_evidence_sha256") or "").lower()
        if len(evidence_hash) != 64 or any(char not in "0123456789abcdef" for char in evidence_hash):
            return False
        available_at = row.get("source_available_at")
        try:
            available_at = datetime.datetime.fromisoformat(str(available_at).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        return available_at.date() < as_of

    @classmethod
    def _snapshot_content_matches(cls, snapshot):
        payload = snapshot.get("snapshot_json")
        if payload is None:
            payload = snapshot
        if not isinstance(payload, dict):
            return False
        expected = payload.get("market_data_content_fingerprint")
        expected_reference = payload.get("security_reference_fingerprint")
        expected_reference_input = payload.get("security_reference_input_fingerprint")
        coverage = payload.get("security_reference_coverage")
        member_count = payload.get("member_count")
        try:
            coverage = float(coverage)
            member_count = int(member_count)
        except (TypeError, ValueError):
            return False
        if (
            not expected
            or payload.get("security_reference_status") != "pass"
            or not 0.999 <= coverage <= 1.0
            or not expected_reference
            or not expected_reference_input
            or member_count < 1
        ):
            return False
        from core.data import StrategyProfileRepo
        references = StrategyProfileRepo.get_security_references(snapshot.get("as_of"))
        members = StrategyProfileRepo.get_snapshot_members(snapshot.get("snapshot_id"))
        member_codes = {row.get("code") for row in members if row.get("code")}
        member_references = [row for row in references if row.get("code") in member_codes]
        try:
            as_of = datetime.date.fromisoformat(str(snapshot.get("as_of"))[:10])
        except (TypeError, ValueError):
            return False
        verified_reference_count = sum(
            cls.security_reference_has_verified_provenance(row, as_of)
            for row in member_references
        )
        current_coverage = verified_reference_count / len(member_codes) if member_codes else 0.0
        if (len(member_codes) != member_count or current_coverage < 0.999
                or abs(current_coverage - coverage) > 1e-9):
            return False
        actual_reference = _security_reference_fingerprint(member_references)
        actual_reference_input = _security_reference_input_fingerprint(
            [row for row in references if row.get("source") == "pit_import"]
        )
        if actual_reference != expected_reference or actual_reference_input != expected_reference_input:
            return False
        cache_key = snapshot.get("snapshot_id")
        cached = cls._snapshot_integrity_cache.get(cache_key)
        if (cached and len(cached) == 5 and cached[0] == expected
                and cached[1] == expected_reference and cached[2] == actual_reference
                and time.monotonic() - cached[3] <= 4 * 60 * 60):
            return cached[4]
        actual = _json_fingerprint(StrategyProfileRepo.market_data_manifest(snapshot["as_of"]))
        matched = actual == expected
        cls._snapshot_integrity_cache[cache_key] = (
            expected, expected_reference, actual_reference, time.monotonic(), matched,
        )
        return matched

    @classmethod
    def build_data_snapshot(cls, as_of, min_history_days=500):
        from core.data import Database, StrategyProfileRepo
        from core.domain.portfolio.strategy_profile import canonical_pool_name

        as_of = datetime.date.fromisoformat(str(as_of)[:10])
        min_history_days = int(min_history_days)
        if min_history_days != 500:
            raise ValueError("稳定历史池当前只允许500个历史观测")
        latest = Database.fetchone("SELECT MAX(trade_date) AS d FROM daily_quotes WHERE trade_date<=%s", (as_of,)) or {}
        latest_date = latest.get("d")
        if latest_date and not isinstance(latest_date, datetime.date):
            latest_date = datetime.date.fromisoformat(str(latest_date)[:10])
        latest_iso = str(latest_date) if latest_date else None
        market_data_manifest = StrategyProfileRepo.market_data_manifest(as_of)
        market_data_content_fingerprint = _json_fingerprint(market_data_manifest)
        imported_references = [
            row for row in StrategyProfileRepo.get_security_references(as_of)
            if row.get("source") == "pit_import"
        ]
        security_reference_input_fingerprint = _security_reference_input_fingerprint(imported_references)
        cached = StrategyProfileRepo.get_snapshot_for_date(as_of, "stable_history_500d")
        if (cached and cached.get("pit_contract_version") == PIT_CONTRACT_VERSION
                and str(cached.get("latest_trade_date"))[:10] == str(latest_iso)
                and cached.get("market_data_content_fingerprint") == market_data_content_fingerprint
                and cached.get("security_reference_input_fingerprint") == security_reference_input_fingerprint):
            return cached
        rows = Database.fetchall(
            "SELECT code,COUNT(*) AS observation_count,MIN(trade_date) AS first_trade_date,MAX(trade_date) AS last_trade_date "
            "FROM daily_quotes WHERE trade_date<%s AND open IS NOT NULL AND close IS NOT NULL "
            "GROUP BY code HAVING COUNT(*)>=%s ORDER BY code",
            (as_of, min_history_days),
        )
        members = []
        for row in rows:
            members.append({
                "code": row["code"],
                "observation_count": int(row["observation_count"]),
                "first_trade_date": str(row["first_trade_date"]),
                "last_trade_date": str(row["last_trade_date"]),
            })
        table_dates = {}
        for table, field in (
            ("daily_quotes", "trade_date"),
            ("daily_basic", "trade_date"),
            ("capital_flow_daily", "trade_date"),
        ):
            try:
                row = Database.fetchone(f"SELECT MAX({field}) AS d FROM {table} WHERE {field}<=%s", (as_of,)) or {}
                table_dates[table] = str(row.get("d")) if row.get("d") else None
            except Exception:
                table_dates[table] = None
        freshness = "fresh"
        if not latest_date or (as_of - latest_date).days > 7:
            freshness = "stale"
        pit_status = "unverified" if members else "fail"
        pit_status_reason = (
            "行情交易日期早于as_of不能证明供应商发布或本系统可见时间；缺少source_available_at证据"
            if members else "as_of之前没有满足历史观测数条件的证券"
        )
        member_codes = {row["code"] for row in members}
        existing_references = {
            row["code"]: row for row in imported_references
            if row.get("code") in member_codes and row.get("source") == "pit_import"
        }
        industry_map = {}
        try:
            sector_rows = Database.fetchall(
                "SELECT code,sector_name FROM stock_sectors WHERE sector_type='industry' "
                "AND updated_at IS NOT NULL AND DATE(updated_at)<=%s",
                (as_of,),
            )
            industry_map = {row["code"]: row.get("sector_name", "") for row in sector_rows if row["code"] in member_codes}
        except Exception:
            industry_map = {}
        volume_map = {}
        if latest_date:
            quote_rows = Database.fetchall(
                "SELECT code,volume FROM daily_quotes WHERE trade_date=%s", (latest_date,),
            )
            volume_map = {row["code"]: row.get("volume") for row in quote_rows if row["code"] in member_codes}
        references = []
        for row in members:
            code = row["code"]
            if code in existing_references:
                references.append(existing_references[code])
                continue
            board = "star" if code.startswith("688") else "chinext" if code.startswith(("300", "301")) else "beijing" if code.startswith(("4", "8")) else "main"
            references.append({
                "code": code,
                "industry": industry_map.get(code, ""),
                "board": board,
                "is_suspended": code not in volume_map or float(volume_map.get(code) or 0) <= 0,
                "listed_date": row.get("first_trade_date"),
                "source": "pit_market_snapshot_partial",
            })
        StrategyProfileRepo.save_security_references(as_of, references)
        authoritative = [row for row in references if cls.security_reference_has_verified_provenance(row, as_of)]
        reference_coverage = len(authoritative) / len(members) if members else 0.0
        reference_status = "pass" if reference_coverage >= 0.999 else "partial"
        reference_fingerprint = _security_reference_fingerprint(references)
        payload = {
            "as_of": as_of.isoformat(),
            "universe_name": canonical_pool_name("stable_history_500d"),
            "min_history_days": min_history_days,
            "latest_trade_date": latest_iso,
            "table_dates": table_dates,
            "members": members,
            "security_reference_fingerprint": reference_fingerprint,
            "security_reference_input_fingerprint": security_reference_input_fingerprint,
            "security_reference_status": reference_status,
            "security_reference_coverage": reference_coverage,
            "pit_contract_version": PIT_CONTRACT_VERSION,
            "pit_status_reason": pit_status_reason,
            "market_data_manifest": market_data_manifest,
            "market_data_content_fingerprint": market_data_content_fingerprint,
        }
        snapshot_id = _json_fingerprint(payload)
        snapshot = {
            "snapshot_id": snapshot_id,
            "as_of": as_of.isoformat(),
            "latest_trade_date": latest_iso,
            "universe_name": "stable_history_500d",
            "member_count": len(members),
            "freshness_status": freshness,
            "pit_status": pit_status,
            "pit_contract_version": PIT_CONTRACT_VERSION,
            "pit_status_reason": pit_status_reason,
            "table_dates": table_dates,
            "min_history_days": min_history_days,
            "security_reference_status": reference_status,
            "security_reference_coverage": reference_coverage,
            "security_reference_fingerprint": reference_fingerprint,
            "security_reference_input_fingerprint": security_reference_input_fingerprint,
            "market_data_manifest": market_data_manifest,
            "market_data_content_fingerprint": market_data_content_fingerprint,
        }
        StrategyProfileRepo.save_snapshot(snapshot, members)
        return snapshot

    @classmethod
    def import_security_references(cls, as_of, references, source_name, source_version,
                                   source_available_at, source_evidence_sha256):
        from core.data import StrategyProfileRepo

        as_of = datetime.date.fromisoformat(str(as_of)[:10])
        source_name = str(source_name or "").strip()
        source_version = str(source_version or "").strip()
        try:
            source_available_at = datetime.datetime.fromisoformat(
                str(source_available_at).strip().replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as error:
            raise ValueError("source_available_at必须是ISO时间") from error
        if not source_name or len(source_name) > 64 or not source_version or len(source_version) > 128:
            raise ValueError("必须提供有效的历史证券来源名称和版本")
        if source_available_at.tzinfo is not None:
            source_available_at = source_available_at.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        if source_available_at.date() >= as_of:
            raise ValueError("历史证券来源必须在决策as_of日期之前已公开")
        source_evidence_sha256 = str(source_evidence_sha256 or "").strip().lower()
        if len(source_evidence_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_evidence_sha256):
            raise ValueError("source_evidence_sha256必须是64位SHA-256")
        normalized = []
        seen = set()
        for row in references or []:
            code = str(row.get("code") or "").strip()
            if len(code) != 6 or not code.isdigit() or code in seen:
                raise ValueError(f"证券代码无效或重复: {code}")
            if not row.get("name") or not row.get("industry") or not row.get("board") or not row.get("listed_date"):
                raise ValueError(f"{code} 缺少名称、行业、板块或上市日期")
            listed_date = datetime.date.fromisoformat(str(row["listed_date"])[:10])
            delisted_date = datetime.date.fromisoformat(str(row["delisted_date"])[:10]) if row.get("delisted_date") else None
            if listed_date > as_of or (delisted_date and delisted_date < listed_date):
                raise ValueError(f"{code} 上市/退市日期无效")
            seen.add(code)
            normalized.append({
                "code": code, "name": str(row["name"]), "industry": str(row["industry"]),
                "board": str(row["board"]), "is_st": bool(row.get("is_st")),
                "is_suspended": bool(row.get("is_suspended")), "listed_date": listed_date,
                "delisted_date": delisted_date, "source": "pit_import",
                "source_name": source_name, "source_version": source_version,
                "source_available_at": source_available_at,
                "source_evidence_sha256": source_evidence_sha256,
                "provenance_status": "unverified",
            })
        if not normalized:
            raise ValueError("历史证券状态快照不能为空")
        StrategyProfileRepo.save_security_references(as_of, normalized)
        return {"as_of": as_of.isoformat(), "count": len(normalized), "status": "imported_unverified"}

    @classmethod
    def create(cls, profile_id, name, config=None, parent_version=None, snapshot_as_of=None, created_by="system"):
        from core.data import StrategyProfileRepo
        from core.domain.portfolio.strategy_profile import normalize_profile_config, profile_fingerprint

        profile_id = str(profile_id or "").strip()
        if not profile_id or len(profile_id) > 64 or not profile_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("profile_id只能包含字母、数字、下划线和连字符，且不超过64字符")
        cfg = normalize_profile_config(config)
        if snapshot_as_of and not cfg["data"].get("snapshot_id"):
            snapshot = cls.build_data_snapshot(snapshot_as_of, cfg["universe"]["min_history_days"])
            cfg["data"]["snapshot_id"] = snapshot["snapshot_id"]
        cfg.setdefault("fingerprints", {})
        from core.adapters.qlib.factor_resolver import resolve_factor_expression
        factor_definitions = []
        for key in cfg["factors"]["keys"]:
            expression, skip_reason = resolve_factor_expression(key)
            factor_definitions.append({"key": key, "expression": expression, "skip_reason": skip_reason})
        cfg["fingerprints"]["factor_definitions"] = _json_fingerprint(factor_definitions)
        cfg["factors"]["definitions"] = factor_definitions
        for ref in cfg["model"].get("shadow_refs") or []:
            definitions = []
            for key in ref.get("factor_keys") or []:
                expression, skip_reason = resolve_factor_expression(key)
                definitions.append({"key": key, "expression": expression, "skip_reason": skip_reason})
            ref["factor_definitions_fingerprint"] = _json_fingerprint(definitions)
        if cfg["model"].get("refs") and cfg["windows"].get("oos_start"):
            from core.adapters.qlib.model_selection import get_model_for_factor_set
            for ref in cfg["model"]["refs"]:
                if ref.get("artifact_fingerprint"):
                    continue
                try:
                    checked = get_model_for_factor_set(
                        ref["experiment_id"], ref["run_id"], cfg["factors"]["keys"],
                        cfg["windows"]["oos_start"], "a500",
                    )
                    ref["artifact_fingerprint"] = checked["artifact_fingerprint"]
                except ValueError:
                    pass
            for ref in cfg["model"].get("shadow_refs") or []:
                if ref.get("artifact_fingerprint"):
                    continue
                try:
                    checked = get_model_for_factor_set(
                        ref["experiment_id"], ref["run_id"], ref.get("factor_keys") or [],
                        cfg["windows"].get("oos_end") or cfg["windows"]["oos_start"], "a500",
                    )
                    ref["artifact_fingerprint"] = checked["artifact_fingerprint"]
                except ValueError:
                    pass
            for fold_ref in cfg["model"].get("oos_schedule") or []:
                if fold_ref.get("artifact_fingerprint") or not fold_ref.get("window_start"):
                    continue
                try:
                    checked = get_model_for_factor_set(
                        fold_ref["experiment_id"], fold_ref["run_id"], cfg["factors"]["keys"],
                        fold_ref["window_start"], "a500", allow_test_window=True,
                    )
                    fold_ref["artifact_fingerprint"] = checked["artifact_fingerprint"]
                    fold_ref.setdefault("train_end", checked.get("train_end"))
                    fold_ref.setdefault("valid_end", checked.get("valid_end"))
                except ValueError:
                    pass
        cfg["fingerprints"]["factor"] = _json_fingerprint({
            "set_name": cfg["factors"]["set_name"],
            "version": cfg["factors"]["version"],
            "keys": cfg["factors"]["keys"],
        })
        cfg["fingerprints"]["model_refs"] = _json_fingerprint({
            "primary": cfg["model"]["refs"],
            "shadow": cfg["model"].get("shadow_refs") or [],
        })
        version = StrategyProfileRepo.next_version(profile_id)
        if parent_version is not None and not StrategyProfileRepo.get(profile_id, int(parent_version)):
            raise ValueError("父版本不存在")
        row = StrategyProfileRepo.create({
            "profile_id": profile_id,
            "version": version,
            "name": str(name or profile_id)[:120],
            "status": "draft",
            "config": cfg,
            "fingerprint": profile_fingerprint(cfg),
            "parent_version": parent_version,
            "created_by": created_by,
        })
        return cls.get(profile_id, version)

    @classmethod
    def list(cls, limit=100):
        from core.data import StrategyProfileRepo
        return [cls._with_evidence(_present(row)) for row in StrategyProfileRepo.list(limit)]

    @classmethod
    def get(cls, profile_id, version):
        from core.data import StrategyProfileRepo
        return cls._with_evidence(_present(StrategyProfileRepo.get(profile_id, int(version))))

    @classmethod
    def champion(cls):
        from core.data import StrategyProfileRepo
        return cls._with_evidence(_present(StrategyProfileRepo.get_champion()))

    @classmethod
    def _with_evidence(cls, profile):
        if not profile:
            return profile
        from core.data import StrategyProfileRepo
        evaluation = StrategyProfileRepo.latest_evaluation(profile["profile_id"], profile["version"])
        paper = StrategyProfileRepo.get_paper_run(profile["profile_id"], profile["version"])
        if paper:
            from core.domain.portfolio.strategy_profile import evaluate_paper_evidence
            periods = StrategyProfileRepo.paper_periods(paper["paper_run_id"])
            paper["periods"] = periods
            paper["evidence"] = evaluate_paper_evidence(periods, cls._champion_excess_calmar(profile))
        profile["latest_evaluation"] = evaluation
        profile["paper_run"] = paper
        profile["runtime_health"] = StrategyProfileRepo.get_runtime_health(profile["profile_id"], profile["version"])
        return profile

    @classmethod
    def validate(cls, profile_id, version, created_by="system"):
        from core.data import StrategyProfileRepo
        from core.domain.portfolio.strategy_profile import assert_single_change, profile_fingerprint, validate_profile_contract

        row = StrategyProfileRepo.get(profile_id, int(version))
        if not row:
            raise ValueError("策略版本不存在")
        if row["status"] not in ("draft", "blocked", "challenger"):
            raise ValueError(f"策略状态 {row['status']} 不允许重新验证")
        cfg = row["config_json"]
        contract = validate_profile_contract(cfg)
        errors = list(contract["errors"])
        warnings = list(contract["warnings"])
        if profile_fingerprint(cfg) != row["fingerprint"]:
            errors.append("Profile内容与持久化指纹不一致")
        if row.get("parent_version") is not None:
            parent = StrategyProfileRepo.get(profile_id, int(row["parent_version"]))
            comparison = assert_single_change(parent["config_json"], cfg) if parent else {"valid": False, "changed_fields": []}
            declared = (cfg.get("experiment") or {}).get("changed_field")
            if not comparison["valid"] or declared not in comparison["changed_fields"]:
                errors.append("Challenger相对父版本必须只改变一个已声明变量")

        snapshot = None
        snapshot_id = cfg["data"].get("snapshot_id")
        if snapshot_id:
            snapshot = StrategyProfileRepo.get_snapshot(snapshot_id)
            if not snapshot:
                errors.append("绑定的数据快照不存在")
            else:
                if not cls.snapshot_has_verified_pit(snapshot):
                    errors.append("数据快照缺少当前PIT契约或未通过时点可见性检查")
                if snapshot["freshness_status"] != "fresh":
                    errors.append("数据快照在绑定时已过期")
                if snapshot["universe_name"] != cfg["universe"]["name"]:
                    errors.append("数据快照股票池与Profile不一致")
                payload = snapshot.get("snapshot_json") or {}
                if payload.get("security_reference_status") != "pass":
                    errors.append("历史行业、ST、上市和停牌状态快照不完整，禁止用于生产准入")
                if not cls._snapshot_content_matches(snapshot):
                    errors.append("绑定数据快照的行情内容已漂移")

        from core.adapters.qlib.factor_resolver import resolve_factor_expression
        current_factor_definitions = []
        for key in cfg["factors"].get("keys") or []:
            expression, skip_reason = resolve_factor_expression(key)
            current_factor_definitions.append({"key": key, "expression": expression, "skip_reason": skip_reason})
            if skip_reason:
                errors.append(f"因子 {key} 无法解析: {skip_reason}")
        if _json_fingerprint(current_factor_definitions) != (cfg.get("fingerprints") or {}).get("factor_definitions"):
            errors.append("因子表达式内容与冻结指纹不一致")

        model_checks = []
        windows = cfg["windows"]
        cutoff = (snapshot or {}).get("latest_trade_date") or windows.get("oos_end")
        from core.adapters.qlib.model_selection import get_model_for_factor_set
        if cutoff and cfg["model"].get("refs"):
            for ref in cfg["model"]["refs"]:
                try:
                    checked = get_model_for_factor_set(
                        ref["experiment_id"], ref["run_id"], cfg["factors"]["keys"], cutoff, "a500",
                    )
                    if int(checked.get("label_horizon") or 0) != 10:
                        errors.append(f"模型 {ref['run_id']} 的label_horizon不是10")
                    if checked.get("factor_keys") != cfg["factors"]["keys"]:
                        errors.append(f"模型 {ref['run_id']} 的因子顺序与Profile不一致")
                    if checked.get("artifact_fingerprint") != ref.get("artifact_fingerprint"):
                        errors.append(f"模型 {ref['run_id']} 的内容指纹已变化")
                    model_checks.append(checked)
                except ValueError as error:
                    errors.append(str(error))
        for ref in cfg["model"].get("shadow_refs") or []:
            current_definitions = []
            for key in ref.get("factor_keys") or []:
                expression, skip_reason = resolve_factor_expression(key)
                current_definitions.append({"key": key, "expression": expression, "skip_reason": skip_reason})
                if skip_reason:
                    errors.append(f"影子因子 {key} 无法解析: {skip_reason}")
            if _json_fingerprint(current_definitions) != ref.get("factor_definitions_fingerprint"):
                errors.append(f"影子模型 {ref['run_id']} 的因子表达式内容已变化")
            try:
                checked = get_model_for_factor_set(
                    ref["experiment_id"], ref["run_id"], ref.get("factor_keys") or [],
                    cutoff or windows.get("oos_end"), "a500",
                )
                if int(checked.get("label_horizon") or 0) != 10:
                    errors.append(f"影子模型 {ref['run_id']} 的label_horizon不是10")
                if checked.get("factor_keys") != ref.get("factor_keys"):
                    errors.append(f"影子模型 {ref['run_id']} 的因子顺序与Profile不一致")
                if checked.get("artifact_fingerprint") != ref.get("artifact_fingerprint"):
                    errors.append(f"影子模型 {ref['run_id']} 的内容指纹已变化")
                model_checks.append({**checked, "shadow": True, "role": ref.get("role")})
            except ValueError as error:
                errors.append(str(error))
        for fold_ref in cfg["model"].get("oos_schedule") or []:
            try:
                checked = get_model_for_factor_set(
                    fold_ref["experiment_id"], fold_ref["run_id"], cfg["factors"]["keys"],
                    fold_ref["window_start"], "a500", allow_test_window=True,
                )
                if checked.get("artifact_fingerprint") != fold_ref.get("artifact_fingerprint"):
                    errors.append(f"OOS折模型 {fold_ref['run_id']} 的内容指纹已变化")
                if checked.get("factor_keys") != cfg["factors"]["keys"]:
                    errors.append(f"OOS折模型 {fold_ref['run_id']} 的因子顺序与Profile不一致")
                for boundary in ("train_start", "train_end", "valid_start", "valid_end"):
                    if str(checked.get(boundary) or "")[:10] != str(fold_ref.get(boundary) or "")[:10]:
                        errors.append(f"OOS折模型 {fold_ref['run_id']} 的{boundary}不一致")
                model_checks.append({**checked, "oos_window": [fold_ref["window_start"], fold_ref["window_end"]]})
            except ValueError as error:
                errors.append(str(error))
        if windows.get("oos_end") and snapshot:
            latest = datetime.date.fromisoformat(str(snapshot["latest_trade_date"])[:10]) if snapshot.get("latest_trade_date") else None
            oos_end = datetime.date.fromisoformat(str(windows["oos_end"])[:10])
            if latest and oos_end > latest:
                errors.append("独立OOS结束日期晚于数据快照可见日期")

        status = "challenger" if not errors else "blocked"
        StrategyProfileRepo.set_status(profile_id, int(version), status, errors, "validate", created_by)
        return {
            "passed": not errors,
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "snapshot": snapshot,
            "models": model_checks,
            "profile": cls.get(profile_id, version),
        }

    @classmethod
    def evaluate(cls, profile_id, version, backtest_job_ids, created_by="system"):
        from core.data import StrategyProfileRepo
        from core.domain.portfolio.strategy_profile import evaluate_oos_evidence
        from core.shared.infrastructure.async_task import task_manager

        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile:
            raise ValueError("策略版本不存在")
        if profile["status"] != "challenger":
            raise ValueError("只有验证通过的Challenger可以提交OOS评估")
        job_ids = list(backtest_job_ids or [])
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("OOS评估任务不得重复")
        folds = []
        for job_id in job_ids:
            task = task_manager.get_dict(job_id)
            if not task or task.get("status") != "completed" or not isinstance(task.get("result"), dict):
                raise ValueError(f"回测任务 {job_id} 尚未成功完成")
            result = task["result"]
            provenance = result.get("strategy_profile") or {}
            if provenance.get("profile_id") != profile_id or int(provenance.get("version") or 0) != int(version):
                raise ValueError(f"回测任务 {job_id} 未绑定当前Profile")
            if provenance.get("fingerprint") != profile["fingerprint"]:
                raise ValueError(f"回测任务 {job_id} 的Profile指纹不一致")
            summary = result.get("summary") or {}
            benchmark = result.get("benchmark_definition") or {}
            if benchmark.get("primary") != "stable_history_500d_equal_weight":
                raise ValueError(f"回测任务 {job_id} 未使用同池等权主基准")
            if benchmark.get("membership_source") != "universe_membership_history":
                raise ValueError(f"回测任务 {job_id} 未使用逐日PIT股票池成员历史")
            run_window = result.get("run_window") or {}
            if not run_window.get("start") or not run_window.get("end"):
                raise ValueError(f"回测任务 {job_id} 缺少不可变OOS窗口")
            expected_fold_model = next((item for item in profile["config_json"]["model"].get("oos_schedule") or []
                                        if item.get("window_start") == str(run_window["start"])[:10]
                                        and item.get("window_end") == str(run_window["end"])[:10]), None)
            if not expected_fold_model or provenance.get("oos_model") != expected_fold_model:
                raise ValueError(f"回测任务 {job_id} 未使用该OOS折冻结的独立模型run")
            max_drawdown = float(summary.get("max_drawdown") or 0)
            annual_excess = float(summary.get("annual_excess") or 0)
            date_excess = [
                float(row.get("period_return") or 0) - float(row.get("benchmark_return") or 0)
                for row in result.get("periods") or []
            ]
            folds.append({
                "job_id": job_id,
                "window_start": str(run_window["start"])[:10],
                "window_end": str(run_window["end"])[:10],
                "annual_excess": annual_excess,
                "max_drawdown": max_drawdown,
                "one_way_turnover": float(summary.get("max_one_way_turnover") or summary.get("avg_one_way_turnover") or 0),
                "excess_calmar": float(summary.get("excess_calmar") or (annual_excess / abs(max_drawdown) if max_drawdown else 0)),
                "date_excess_returns": date_excess,
                "decision_dates": [str(row.get("date") or row.get("decision_date") or "")[:10] for row in result.get("periods") or []],
                "benchmark_coverage": min(
                    (float(row.get("benchmark_coverage") or 0) for row in result.get("periods") or []), default=0,
                ),
                "model_ref": {key: expected_fold_model.get(key) for key in ("experiment_id", "run_id", "artifact_fingerprint")},
            })
        evaluation = evaluate_oos_evidence(folds, cls._champion_excess_calmar(profile))
        evaluation_key = _json_fingerprint({"profile": profile["fingerprint"], "jobs": sorted(job_ids)})
        saved = StrategyProfileRepo.save_evaluation(profile_id, int(version), evaluation_key, evaluation)
        StrategyProfileRepo.add_transition(profile_id, int(version), "challenger", "challenger", "evaluate", {
            "passed": evaluation["passed"], "failures": evaluation["failures"], "evaluation_key": evaluation_key,
        }, created_by)
        return {"evaluation": saved, "result": evaluation}

    @classmethod
    def start_paper(cls, profile_id, version, created_by="system"):
        from core.data import StrategyProfileRepo

        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile:
            raise ValueError("策略版本不存在")
        evaluation = StrategyProfileRepo.latest_evaluation(profile_id, int(version))
        if profile["status"] != "challenger" or not evaluation or not evaluation.get("passed"):
            raise ValueError("必须先通过Challenger OOS评估")
        paper = StrategyProfileRepo.start_paper(uuid.uuid4().hex, profile_id, int(version))
        StrategyProfileRepo.set_status(profile_id, int(version), "paper", [], "paper_start", created_by)
        return cls.get(profile_id, version)

    @classmethod
    def record_paper_period(cls, profile_id, version, period, created_by="system"):
        from core.data import Database, PortfolioDecisionRepo, ScreeningSaveRepo, StrategyProfileRepo
        from core.domain.backtest.common import limit_pct_for_code

        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile or profile["status"] != "paper":
            raise ValueError("只有Paper状态可以记录模拟盘周期")
        paper = StrategyProfileRepo.get_paper_run(profile_id, int(version))
        if not paper:
            raise ValueError("模拟盘任务不存在")

        save = ScreeningSaveRepo.get_by_id(int(period["screening_save_id"]))
        decision = PortfolioDecisionRepo.get_run(str(period["decision_id"]))
        if not save or not decision:
            raise ValueError("选股快照或决策记录不存在")
        expected_ref = (profile_id, int(version), profile["fingerprint"])
        save_ref = (save.get("profile_id"), int(save.get("profile_version") or 0), save.get("profile_fingerprint"))
        decision_ref = (
            decision.get("profile_id"), int(decision.get("profile_version") or 0), decision.get("profile_fingerprint"),
        )
        if save_ref != expected_ref or decision_ref != expected_ref:
            raise ValueError("模拟盘证据与Profile版本或指纹不一致")
        if int(decision.get("screening_save_id") or 0) != int(save["id"]):
            raise ValueError("决策记录与选股快照不匹配")
        if decision.get("policy_id") != "legacy":
            raise ValueError("Paper准入只接受纯排名基线决策")

        decision_date = datetime.date.fromisoformat(str(decision["eval_date"])[:10])
        started_at = datetime.date.fromisoformat(str(paper["started_at"])[:10])
        if decision_date < started_at:
            raise ValueError("不能用Paper启动前的历史决策回填20期")
        dates = Database.fetchall(
            "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date>%s ORDER BY trade_date LIMIT 11",
            (decision_date,),
        )
        if len(dates) < 11:
            raise ValueError("T+1开盘到T+11开盘的持有收益尚未成熟")
        entry_date = dates[0]["trade_date"]
        exit_date = dates[10]["trade_date"]
        pre_exit_date = dates[9]["trade_date"]
        today = datetime.date.today()
        if today < exit_date:
            raise ValueError("T+1开盘到T+11开盘的持有收益尚未成熟")

        result = decision.get("result_json") or {}
        selected = [row.get("code") for row in result.get("selected") or [] if row.get("code")]
        top_n = int(profile["config_json"]["selection"]["top_n"])
        if len(selected) != top_n:
            raise ValueError(f"决策选中数量不是Profile锁定的Top{top_n}")
        candidates = [row.get("code") for row in save.get("candidates") or [] if row.get("code")]
        signal_consistency = 1.0 if selected == candidates[:top_n] else 0.0

        entry_references = {
            row["code"]: row for row in StrategyProfileRepo.get_security_references(entry_date)
            if row.get("source") == "pit_import"
        }
        exit_references = {
            row["code"]: row for row in StrategyProfileRepo.get_security_references(exit_date)
            if row.get("source") == "pit_import"
        }

        placeholders = ",".join(["%s"] * len(selected))
        price_rows = Database.fetchall(
            f"SELECT code,trade_date,open,close,volume,adj_factor FROM daily_quotes "
            f"WHERE code IN ({placeholders}) AND trade_date IN (%s,%s,%s,%s)",
            tuple(selected) + (decision_date, entry_date, pre_exit_date, exit_date),
        )
        prices = {(row["code"], str(row["trade_date"])[:10]): row for row in price_rows}
        config = profile["config_json"]
        execution = config["execution"]
        buy_cost = float(execution["commission_rate"]) + float(execution["slippage_rate"])
        sell_cost = buy_cost + float(execution["stamp_duty"])
        weight = float(config["portfolio"]["target_exposure"]) / top_n
        portfolio_return = 0.0
        fills = []
        invalid_reasons = []
        for code in selected:
            entry_reference = entry_references.get(code)
            exit_reference = exit_references.get(code)
            if not entry_reference or not exit_reference:
                invalid_reasons.append(f"{code} 缺少T+1或T+11历史证券状态")
                continue
            if entry_reference.get("is_suspended"):
                fills.append({"code": code, "status": "unfilled", "reason": "T+1停牌", "weight": weight})
                continue
            if exit_reference.get("is_suspended"):
                invalid_reasons.append(f"{code} T+11停牌无法卖出")
                continue
            if entry_reference.get("listed_date") and str(entry_reference["listed_date"])[:10] > str(entry_date)[:10]:
                invalid_reasons.append(f"{code} T+1尚未上市")
                continue
            if exit_reference.get("delisted_date") and str(exit_reference["delisted_date"])[:10] <= str(exit_date)[:10]:
                invalid_reasons.append(f"{code} T+11已退市")
                continue
            previous = prices.get((code, decision_date.isoformat()))
            entry = prices.get((code, str(entry_date)[:10]))
            pre_exit = prices.get((code, str(pre_exit_date)[:10]))
            exit_row = prices.get((code, str(exit_date)[:10]))
            if not previous or not entry or not pre_exit or not exit_row:
                invalid_reasons.append(f"{code} 缺少决策/T+1/T+11价格")
                continue
            prev_close = float(previous.get("close") or 0)
            entry_open = float(entry.get("open") or 0)
            exit_open = float(exit_row.get("open") or 0)
            if min(prev_close, entry_open, exit_open) <= 0 or float(entry.get("volume") or 0) <= 0 or float(exit_row.get("volume") or 0) <= 0:
                invalid_reasons.append(f"{code} 停牌或价格无效")
                continue
            entry_threshold = limit_pct_for_code(code, is_st=bool(entry_reference.get("is_st"))) * 0.98
            entry_return = entry_open / prev_close
            if previous.get("adj_factor") and entry.get("adj_factor"):
                entry_return *= float(previous["adj_factor"]) / float(entry["adj_factor"])
            entry_return -= 1.0
            if entry_return >= entry_threshold:
                fills.append({"code": code, "status": "unfilled", "reason": "T+1开盘涨停", "weight": weight})
                continue
            pre_exit_close = float(pre_exit.get("close") or 0)
            exit_return = exit_open / pre_exit_close if pre_exit_close > 0 else 0
            if pre_exit.get("adj_factor") and exit_row.get("adj_factor"):
                exit_return *= float(pre_exit["adj_factor"]) / float(exit_row["adj_factor"])
            exit_return -= 1.0
            exit_threshold = limit_pct_for_code(code, is_st=bool(exit_reference.get("is_st"))) * 0.98
            if exit_return <= -exit_threshold:
                invalid_reasons.append(f"{code} T+11开盘跌停无法卖出")
                continue
            gross = exit_open / entry_open - 1.0
            net = (1.0 + gross) * (1.0 - buy_cost) * (1.0 - sell_cost) - 1.0
            portfolio_return += weight * net
            fills.append({
                "code": code, "status": "filled", "entry_date": str(entry_date), "exit_date": str(exit_date),
                "entry_price": entry_open, "exit_price": exit_open, "gross_return": gross,
                "net_return": net, "weight": weight,
            })

        snapshot_id = save.get("data_fingerprint")
        snapshot = StrategyProfileRepo.get_snapshot(snapshot_id) if snapshot_id else None
        if (not cls.snapshot_has_current_integrity(snapshot)
                or snapshot.get("freshness_status") != "fresh"
                or snapshot.get("security_reference_status") != "pass"):
            raise ValueError("选股快照未绑定通过PIT、新鲜度与历史证券状态检查的当期数据快照")
        if str(snapshot.get("as_of"))[:10] != decision_date.isoformat():
            raise ValueError("选股数据快照日期与Paper决策日期不一致")
        members = StrategyProfileRepo.get_snapshot_members(snapshot_id)
        member_codes = [row["code"] for row in members]
        benchmark_returns = []
        for offset in range(0, len(member_codes), 500):
            batch = member_codes[offset:offset + 500]
            marks = ",".join(["%s"] * len(batch))
            rows = Database.fetchall(
                f"SELECT code,trade_date,open FROM daily_quotes WHERE code IN ({marks}) AND trade_date IN (%s,%s)",
                tuple(batch) + (entry_date, exit_date),
            )
            by_code = {}
            for row in rows:
                by_code.setdefault(row["code"], {})[str(row["trade_date"])[:10]] = float(row.get("open") or 0)
            for code, values in by_code.items():
                entry_open = values.get(str(entry_date)[:10], 0)
                exit_open = values.get(str(exit_date)[:10], 0)
                if entry_open > 0 and exit_open > 0:
                    benchmark_returns.append(exit_open / entry_open - 1.0)
        benchmark_return = sum(benchmark_returns) / len(benchmark_returns) if benchmark_returns else 0.0

        prior = [row for row in StrategyProfileRepo.paper_periods(paper["paper_run_id"]) if row.get("matured") and row.get("valid")]
        prior_codes = set(prior[-1].get("selected_codes") or []) if prior else set()
        one_way_turnover = 0.0 if not prior else float(config["portfolio"]["target_exposure"]) * (1 - len(prior_codes & set(selected)) / top_n)
        equity = 1.0
        peak = 1.0
        drawdown = 0.0
        for row in prior:
            equity *= 1.0 + float(row.get("portfolio_return") or 0)
            peak = max(peak, equity)
        equity *= 1.0 + portfolio_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        valid = not invalid_reasons and bool(benchmark_returns)
        computed = {
            "decision_date": decision_date.isoformat(), "entry_date": str(entry_date), "exit_date": str(exit_date),
            "matured": True, "valid": valid, "profile_fingerprint": profile["fingerprint"],
            "portfolio_return": portfolio_return, "benchmark_return": benchmark_return,
            "excess_return": portfolio_return - benchmark_return, "drawdown": drawdown,
            "one_way_turnover": one_way_turnover, "signal_consistency": signal_consistency,
            "screening_save_id": int(save["id"]), "decision_id": decision["decision_id"],
            "data_fingerprint": snapshot_id,
            "selected_codes": selected, "fills": fills,
            "invalid_reason": "; ".join(invalid_reasons) if invalid_reasons else None,
        }
        StrategyProfileRepo.save_paper_period(paper["paper_run_id"], decision_date, computed)
        StrategyProfileRepo.add_transition(profile_id, int(version), "paper", "paper", "paper_period", {
            "decision_date": decision_date.isoformat(), "valid": valid, "matured": True,
        }, created_by)
        return cls.get(profile_id, version)

    @classmethod
    def sync_paper_periods(cls, profile_id, version, created_by="system"):
        from core.data import StrategyProfileRepo

        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile or profile["status"] != "paper":
            raise ValueError("只有Paper状态可以同步模拟盘周期")
        paper = StrategyProfileRepo.get_paper_run(profile_id, int(version))
        if not paper:
            raise ValueError("模拟盘任务不存在")
        existing = {
            str(row.get("decision_date"))[:10]
            for row in StrategyProfileRepo.paper_periods(paper["paper_run_id"])
        }
        recorded = []
        pending = []
        for candidate in StrategyProfileRepo.paper_candidates(profile_id, int(version), paper["started_at"]):
            decision_date = str(candidate.get("eval_date"))[:10]
            if decision_date in existing:
                continue
            try:
                cls.record_paper_period(profile_id, int(version), {
                    "screening_save_id": int(candidate["screening_save_id"]),
                    "decision_id": candidate["decision_id"],
                }, created_by)
                recorded.append(decision_date)
                existing.add(decision_date)
            except ValueError as error:
                pending.append({"decision_date": decision_date, "reason": str(error)})
        return {"recorded": recorded, "pending": pending, "profile": cls.get(profile_id, version)}

    @classmethod
    def activate(cls, profile_id, version, created_by="system"):
        from core.data import StrategyProfileRepo
        from core.domain.portfolio.strategy_profile import evaluate_paper_evidence

        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile or profile["status"] != "paper":
            raise ValueError("只有Paper状态可以激活为Champion")
        evaluation = StrategyProfileRepo.latest_evaluation(profile_id, int(version))
        paper = StrategyProfileRepo.get_paper_run(profile_id, int(version))
        periods = StrategyProfileRepo.paper_periods(paper["paper_run_id"]) if paper else []
        paper_evidence = evaluate_paper_evidence(periods, cls._champion_excess_calmar(profile))
        failures = []
        if not evaluation or not evaluation.get("passed"):
            failures.append("OOS评估未通过")
        failures.extend(paper_evidence["failures"])
        failures.extend(cls.runtime_health_failures(_present(profile)))
        if failures:
            return {"activated": False, "failures": failures, "paper_evidence": paper_evidence}
        activated = StrategyProfileRepo.activate(profile_id, int(version), "activate", created_by)
        return {"activated": True, "profile": cls._with_evidence(_present(activated)), "paper_evidence": paper_evidence}

    @classmethod
    def rollback(cls, profile_id, version, created_by="system"):
        from core.data import StrategyProfileRepo
        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile or profile["status"] != "superseded":
            raise ValueError("只能回滚到曾经激活且当前为superseded的版本")
        transitions = StrategyProfileRepo.transitions(profile_id, int(version))
        if not any(row["to_status"] == "champion" for row in transitions):
            raise ValueError("目标版本从未成为Champion")
        return cls._with_evidence(_present(StrategyProfileRepo.activate(profile_id, int(version), "rollback", created_by)))

    @classmethod
    def retire(cls, profile_id, version, created_by="system"):
        from core.data import StrategyProfileRepo
        profile = StrategyProfileRepo.get(profile_id, int(version))
        if not profile:
            raise ValueError("策略版本不存在")
        if profile.get("is_active"):
            raise ValueError("当前Champion不能直接退役，请先激活或回滚其他版本")
        return cls._with_evidence(_present(StrategyProfileRepo.set_status(
            profile_id, int(version), "retired", [], "retire", created_by,
        )))

    @classmethod
    def resolve(cls, profile_id=None, version=None, require_status=None):
        if not profile_id:
            profile = cls.champion()
            if not profile:
                return None
        else:
            if version is None:
                raise ValueError("指定profile_id时必须同时指定profile_version")
            profile = cls.get(profile_id, int(version))
            if not profile:
                raise ValueError("策略版本不存在")
        if require_status and profile["status"] not in require_status:
            raise ValueError(f"策略状态 {profile['status']} 不允许当前操作")
        return profile

    @classmethod
    def model_ref_for_date(cls, profile, decision_date, require_schedule=False):
        config = profile.get("config") or profile.get("config_json") or {}
        decision_date = str(decision_date)[:10]
        matches = [
            item for item in (config.get("model") or {}).get("oos_schedule") or []
            if str(item.get("window_start") or "")[:10] <= decision_date
            <= str(item.get("window_end") or "")[:10]
        ]
        if len(matches) > 1:
            raise ValueError("决策日同时命中多个样本外模型窗口")
        if matches:
            return matches[0], True
        if require_schedule:
            raise ValueError("决策日未匹配策略版本冻结的样本外模型窗口")
        refs = (config.get("model") or {}).get("refs") or []
        if not refs:
            raise ValueError("策略版本未绑定主模型")
        return refs[0], False

    @classmethod
    def bind_operation(cls, profile, operation, explicit_overrides=None):
        from core.domain.portfolio.strategy_profile import operation_config
        locked = operation_config(profile["config"], operation)
        conflicts = {}
        for key, value in (explicit_overrides or {}).items():
            if key in locked and value is not None and value != locked[key]:
                conflicts[key] = {"profile": locked[key], "request": value}
        if conflicts:
            raise ValueError("请求参数与不可变Profile冲突: " + ", ".join(conflicts))
        return locked

    @classmethod
    def ensure_execution_allowed(cls, profile_id, version, fingerprint, enforce=True):
        if not enforce:
            return True
        champion = cls.champion()
        if not champion:
            raise ValueError("当前没有Champion，禁止生成实盘新增买入信号")
        if not profile_id or version is None:
            raise ValueError("已有Champion，旧版Legacy快照不可生成新增买入信号")
        if (champion["profile_id"], int(champion["version"])) != (profile_id, int(version)):
            raise ValueError("只有当前Champion可以生成实盘新增买入信号")
        if fingerprint != champion["fingerprint"]:
            raise ValueError("选股快照与Champion指纹不一致")
        failures = cls.runtime_health_failures(champion)
        if failures:
            raise ValueError("；".join(failures))
        return True

    @classmethod
    def runtime_health_failures(cls, champion):
        from core.data import StrategyProfileRepo
        from core.adapters.qlib.model_selection import get_model_for_factor_set
        failures = []
        config = champion.get("config") or {}
        snapshot_id = (config.get("data") or {}).get("snapshot_id")
        snapshot = StrategyProfileRepo.get_snapshot(snapshot_id) if snapshot_id else None
        if not snapshot:
            failures.append("Champion绑定的数据快照不存在，禁止新增买入")
        else:
            if not cls.snapshot_has_verified_pit(snapshot):
                failures.append("Champion绑定的数据快照缺少当前PIT契约或时点可见性验证，禁止新增买入")
            if not cls._snapshot_content_matches(snapshot):
                failures.append("Champion数据内容发生漂移，禁止新增买入")
        runtime_health = StrategyProfileRepo.get_runtime_health(champion["profile_id"], champion["version"])
        if runtime_health and int(runtime_health.get("drawdown_breached") or 0):
            failures.append(
                f"Champion实盘最大回撤已越过-20%（{float(runtime_health.get('max_drawdown') or 0):.2%}），禁止新增买入"
            )
        for ref in (config.get("model") or {}).get("refs") or []:
            try:
                checked = get_model_for_factor_set(
                    ref["experiment_id"], ref["run_id"], (config.get("factors") or {}).get("keys") or [],
                    datetime.date.today().isoformat(), "a500",
                )
                if checked.get("artifact_fingerprint") != ref.get("artifact_fingerprint"):
                    failures.append(f"Champion模型 {ref['run_id']} 内容发生漂移，禁止新增买入")
            except ValueError as error:
                failures.append(f"Champion模型不可用: {error}")
        return failures

    @classmethod
    def _champion_excess_calmar(cls, excluding=None):
        from core.data import StrategyProfileRepo
        champion = StrategyProfileRepo.get_champion()
        if not champion:
            return None
        if excluding and (champion["profile_id"], int(champion["version"])) == (excluding["profile_id"], int(excluding["version"])):
            return None
        evaluation = StrategyProfileRepo.latest_evaluation(champion["profile_id"], champion["version"])
        metrics = (evaluation or {}).get("metrics_json") or {}
        return metrics.get("excess_calmar")
