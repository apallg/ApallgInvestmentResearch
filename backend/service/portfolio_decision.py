import datetime
import json
import logging
import math
import statistics
import uuid


logger = logging.getLogger(__name__)


def load_fundamental_quality(codes, as_of):
    from core.data.connection import Database

    if not codes:
        return {}
    placeholders = ",".join(["%s"] * len(codes))
    rows = Database.fetchall(
        f"SELECT code,report_date,ann_date,source_observed_at,source_capture_id,source_record_hash,"
        f"source_name,version_kind,payload_json FROM fundamental_data_versions "
        f"WHERE ann_date IS NOT NULL AND ann_date<=%s AND DATE(source_observed_at)<%s "
        f"AND code IN ({placeholders}) AND ("
        f"(source_name='tushare_fina_indicator' AND version_kind='provider_observation') OR "
        f"(source_name='official_report' AND version_kind='official_field_reconciliation')) "
        f"ORDER BY code,report_date,ann_date,source_observed_at,source_name,source_capture_id,source_record_hash",
        (as_of, as_of) + tuple(codes),
    )
    version_groups = {}
    for row in rows:
        code = row.get("code")
        report_date = str(row.get("report_date") or "")[:10]
        ann_date = str(row.get("ann_date") or "")[:10]
        if code and report_date and ann_date:
            version_groups.setdefault((code, report_date, ann_date), []).append(row)

    latest_groups = {}
    for group_key, versions in version_groups.items():
        code, report_date, ann_date = group_key
        providers = [row for row in versions if row.get("version_kind") == "provider_observation"]
        if not providers:
            continue
        provider = max(providers, key=lambda row: (
            str(row.get("source_observed_at") or ""),
            str(row.get("source_capture_id") or ""),
            str(row.get("source_record_hash") or ""),
        ))
        group_key = (report_date, ann_date)
        if code not in latest_groups or group_key > latest_groups[code][0]:
            latest_groups[code] = (group_key, provider, versions)

    result = {}
    for code, (_, provider, versions) in latest_groups.items():
        try:
            payload = json.loads(provider.get("payload_json"))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        quality = {
            "profit_dedt": payload.get("profit_dedt"),
            "debt_to_assets": payload.get("debt_to_assets"),
        }
        corrections = sorted(
            (row for row in versions if row.get("version_kind") == "official_field_reconciliation"),
            key=lambda row: (str(row.get("source_observed_at") or ""),
                             str(row.get("source_capture_id") or ""),
                             str(row.get("source_record_hash") or "")),
        )
        for correction in corrections:
            try:
                correction_payload = json.loads(correction.get("payload_json"))
            except (TypeError, ValueError):
                quality = None
                break
            official_values = correction_payload.get("official_values") if isinstance(correction_payload, dict) else None
            if not isinstance(official_values, dict):
                quality = None
                break
            for field in quality:
                if field in official_values:
                    quality[field] = official_values[field]
        if quality is None:
            continue
        for field, value in quality.items():
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            quality[field] = value if value is not None and math.isfinite(value) else None
        result[code] = quality
    try:
        from core.data.fetcher import DataFetcher
        industry_map = DataFetcher().get_industry_map() or {}
        for code, quality in result.items():
            info = industry_map.get(code, {})
            if isinstance(info, dict):
                quality["industry"] = info.get("industry", "")
                quality["name"] = info.get("name", "")
    except Exception as error:
        logger.warning("基本面质量过滤未能加载行业分类: %s", error)
    return result


def load_amount_median_20d(codes, as_of):
    from core.data.connection import Database

    if not codes:
        return {}
    placeholders = ",".join(["%s"] * len(codes))
    params = tuple(codes) + (as_of,)
    rows = Database.fetchall(
        f"SELECT code,amount FROM ("
        f"SELECT code,amount,ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) row_num "
        f"FROM daily_quotes WHERE code IN ({placeholders}) AND trade_date<=%s"
        f") ranked WHERE row_num<=20",
        params,
    )
    grouped = {}
    for row in rows:
        amount = row.get("amount")
        if amount is not None and float(amount) > 0:
            grouped.setdefault(row["code"], []).append(float(amount))
    return {
        code: float(statistics.median(values))
        for code, values in grouped.items()
        if len(values) >= 15
    }


def load_recent_amount(codes, as_of):
    return load_amount_median_20d(codes, as_of)


class PortfolioDecisionService:
    @classmethod
    def _policy(cls, policy_id):
        from core.data import PortfolioDecisionRepo
        from core.domain.portfolio.decision import LEGACY_POLICY, SHADOW_V1_POLICY, SHADOW_V2_POLICY

        builtins = {"legacy": LEGACY_POLICY, "shadow_v1": SHADOW_V1_POLICY, "shadow_v2": SHADOW_V2_POLICY}
        if policy_id in builtins:
            policy = dict(builtins[policy_id])
            policy["status"] = "active" if policy_id == "legacy" else "shadow"
            PortfolioDecisionRepo.upsert_policy(policy, policy["status"])
            return policy
        row = PortfolioDecisionRepo.get_policy(policy_id)
        if not row:
            raise ValueError(f"决策策略不存在: {policy_id}")
        policy = dict(row["policy_json"])
        policy["status"] = row["status"]
        return policy

    @classmethod
    def preview(cls, screening_save_id, top_n=5, policy_id="legacy", selected_codes=None,
                profile_id=None, profile_version=None):
        from core.data import HoldingsRepo, PortfolioDecisionRepo, ScreeningSaveRepo
        from core.domain.portfolio.decision import (
            build_candidate_decision,
            build_shadow_overlays,
            build_shadow_v2_overlays,
            decision_fingerprint,
        )
        from core.domain.screening.exclusion import is_excluded, prefilter_by_liquidity_capacity, prefilter_by_quality
        from core.shared.trading_time import get_latest_trading_day

        save = ScreeningSaveRepo.get_by_id(screening_save_id)
        if not save:
            return {"error": f"选股保存 #{screening_save_id} 不存在"}
        if profile_id:
            if (save.get("profile_id"), int(save.get("profile_version") or 0)) != (profile_id, int(profile_version or 0)):
                return {"error": "决策请求Profile与选股保存不一致"}
        candidates = list(save.get("candidates") or [])
        if selected_codes:
            selected_set = set(selected_codes)
            candidates = [candidate for candidate in candidates if candidate.get("code") in selected_set]
        if not candidates:
            return {"error": "决策候选为空"}

        try:
            policy = cls._policy(policy_id)
        except ValueError as error:
            return {"error": str(error)}

        data_cutoff = get_latest_trading_day()
        codes = [candidate.get("code") for candidate in candidates if candidate.get("code")]
        try:
            amounts = load_amount_median_20d(codes, data_cutoff)
        except Exception as error:
            logger.warning("流动性数据不可用，候选将被执行容量门禁阻断: %s", error)
            amounts = {}
        try:
            quality = load_fundamental_quality(codes, data_cutoff)
        except Exception as error:
            logger.warning("基本面质量数据不可用，候选将被质量门禁阻断: %s", error)
            quality = {}
        structural_dropped = []
        structural_kept = []
        names = {candidate.get("code"): candidate.get("name", "") for candidate in candidates}
        for code, values in quality.items():
            values["name"] = names.get(code) or values.get("name", "")
        reason_labels = {
            "st_stock": "ST股票",
            "beijing_exchange": "北交所股票不在当前交易范围",
        }
        for code in codes:
            excluded, reason = is_excluded(code, names.get(code, ""))
            if excluded:
                structural_dropped.append((code, reason_labels.get(reason, reason)))
            else:
                structural_kept.append(code)
        kept, dropped_quality = prefilter_by_quality(structural_kept, quality, require_complete=True)
        _, dropped_amount, _ = prefilter_by_liquidity_capacity(kept, amounts)
        dropped = structural_dropped + dropped_quality + dropped_amount

        dropped_map = {code: reason for code, reason in dropped}
        unknown_quality_codes = {
            code for code, reason in dropped_quality
            if "状态未知" in reason or "数据缺失" in reason
        }
        enriched = []
        for candidate in candidates:
            row = dict(candidate)
            row["median_amount_20d"] = amounts.get(row.get("code"))
            row["recent_amount"] = amounts.get(row.get("code"))
            row["quality_status"] = "unknown" if row.get("code") in unknown_quality_codes else "available"
            if row.get("code") in dropped_map:
                row["hard_excluded"] = True
                row["exclusion_reason"] = dropped_map[row["code"]]
            enriched.append(row)

        source_candidates = [dict(candidate) for candidate in enriched]
        strategy_audit = None
        if policy_id == "shadow_v1":
            enriched = build_shadow_overlays(enriched)
        elif policy.get("score_mode") == "rank_overlay_v2":
            enriched, strategy_audit = build_shadow_v2_overlays(enriched, policy)

        holdings = HoldingsRepo.get_all()
        current_codes = {row.get("code") for row in holdings if int(row.get("shares") or 0) > 0}
        result = build_candidate_decision(enriched, top_n, policy, current_codes)
        result["quality_gate"] = {
            "status": "blocked" if unknown_quality_codes else "passed",
            "unknown_codes": sorted(unknown_quality_codes),
            "data_cutoff": data_cutoff.isoformat(),
        }
        if strategy_audit:
            result["strategy_audit"] = strategy_audit
        source = {
            "screening_save_id": screening_save_id,
            "screening_eval_date": str(save.get("eval_date")),
            "screening_params": save.get("params_json"),
            "data_cutoff": data_cutoff.isoformat(),
            "candidates": source_candidates,
        }
        decision_id = uuid.uuid4().hex
        result.update({
            "decision_id": decision_id,
            "screening_save_id": screening_save_id,
            "screening_eval_date": str(save.get("eval_date")),
            "eval_date": data_cutoff.isoformat(),
            "data_cutoff": data_cutoff.isoformat(),
            "source_fingerprint": decision_fingerprint(source),
        })
        if save.get("profile_id"):
            result["strategy_profile"] = {
                "profile_id": save["profile_id"],
                "version": int(save["profile_version"]),
                "fingerprint": save.get("profile_fingerprint"),
                "data_fingerprint": save.get("data_fingerprint"),
                "model_fingerprint": save.get("model_fingerprint"),
                "factor_fingerprint": save.get("factor_fingerprint"),
            }
        else:
            result["strategy_profile"] = {"legacy": True, "promotable": False}
        run = {
            "decision_id": decision_id,
            "screening_save_id": screening_save_id,
            "eval_date": data_cutoff.isoformat(),
            "data_cutoff": data_cutoff.isoformat(),
            "policy_id": result["summary"]["policy_id"],
            "policy_version": result["summary"]["policy_version"],
            "mode": result["summary"]["mode"],
            "top_n": top_n,
            "source_fingerprint": result["source_fingerprint"],
            "result_fingerprint": result["fingerprint"],
            "request": {"selected_codes": selected_codes},
            "result": result,
            "profile_id": save.get("profile_id"),
            "profile_version": save.get("profile_version"),
            "profile_fingerprint": save.get("profile_fingerprint"),
            "data_fingerprint": save.get("data_fingerprint"),
            "model_fingerprint": save.get("model_fingerprint"),
            "factor_fingerprint": save.get("factor_fingerprint"),
        }
        PortfolioDecisionRepo.save_run(run, result["ranked_candidates"])
        return result

    @classmethod
    def get(cls, decision_id):
        from core.data import PortfolioDecisionRepo

        row = PortfolioDecisionRepo.get_run(decision_id)
        if not row:
            return None
        result = row["result_json"]
        result["created_at"] = row.get("created_at").isoformat() if isinstance(
            row.get("created_at"), (datetime.date, datetime.datetime)
        ) else row.get("created_at")
        return result
