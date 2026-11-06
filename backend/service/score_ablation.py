import gzip
import hashlib
import json
import logging
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from backend.service.qlib_access import qlib_read_locked

from core.data import Database
from core.data.fetcher import DataFetcher
from core.data.repos.quotes import DailyQuotesRepo
from core.domain.backtest.common import is_open_limit, limit_pct_for_code, norm_code
from core.domain.screening._screener import normalize_qlib_scores
from core.domain.screening.exclusion import evaluate_decision_day_filter
from core.domain.screening.macro import apply_macro_sector_bias
from core.domain.screening.score_pipeline import score_fingerprint
from core.domain.screening.score_replay import replay_score_variants
from core.shared import config


DEFAULT_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")

logger = logging.getLogger(__name__)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_signature(model_id):
    if not model_id or "/" not in str(model_id):
        return None
    experiment_id, run_id = str(model_id).split("/", 1)
    root = Path(str(config.QLIB_MLFLOW_URI).replace("file:", ""))
    artifacts = root / experiment_id / run_id / "artifacts"
    paths = [artifacts / "params.pkl", artifacts / "params_lstm.pkl", artifacts / "lstm_bundle.pkl"]
    if not all(path.exists() for path in paths):
        return None
    return hashlib.sha256("".join(_sha256(path) for path in paths).encode("ascii")).hexdigest()


def _stable_hash(value):
    raw = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=True,
                     separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "core" / "domain" / "screening" / "score_replay.py",
        root / "core" / "domain" / "screening" / "score_pipeline.py",
        root / "core" / "domain" / "screening" / "exclusion.py",
        root / "core" / "domain" / "screening" / "macro.py",
        root / "core" / "domain" / "backtest" / "common.py",
    )
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in paths
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_prediction_cache(path, expected_model_id=None, trusted=False):
    path = Path(path).resolve()
    cache_root = (
        Path(str(config.QLIB_PROVIDER_URI).replace("file:", ""))
        / "prediction_cache"
    ).resolve()
    if not trusted and not path.is_relative_to(cache_root):
        raise ValueError(f"预测缓存必须位于内部目录: {cache_root}")
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    prediction = payload.get("prediction") if isinstance(payload, dict) else payload
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError("预测缓存必须只有一列 prediction")
        prediction = prediction.iloc[:, 0]
    if not isinstance(prediction, pd.Series) or not isinstance(prediction.index, pd.MultiIndex):
        raise ValueError("预测缓存必须是 MultiIndex Series")
    names = list(prediction.index.names)
    date_level = names.index("datetime") if "datetime" in names else 0
    instrument_level = names.index("instrument") if "instrument" in names else 1
    frame = pd.DataFrame({
        "date": pd.to_datetime(prediction.index.get_level_values(date_level), errors="coerce").normalize(),
        "code": [norm_code(value) for value in prediction.index.get_level_values(instrument_level)],
        "score": pd.to_numeric(prediction.to_numpy(), errors="coerce"),
    }).dropna(subset=["date", "score"])
    frame = frame[frame["code"].str.len() == 6]
    if frame.empty:
        raise ValueError("预测缓存没有有效记录")
    embedded = dict(payload.get("metadata") or {}) if isinstance(payload, dict) else {}
    components = payload.get("components") if isinstance(payload, dict) else None
    valid_components = sorted(
        str(key) for key, value in (components or {}).items()
        if isinstance(value, pd.Series) and isinstance(value.index, pd.MultiIndex)
    )
    embedded_model_id = embedded.get("model_id")
    artifact_signature = embedded.get("artifact_signature")
    if expected_model_id and embedded_model_id and str(expected_model_id) != str(embedded_model_id):
        raise ValueError(
            f"预测缓存 model_id 不匹配: expected={expected_model_id}, embedded={embedded_model_id}")
    current_signature = _model_artifact_signature(embedded_model_id)
    if current_signature and artifact_signature and current_signature != artifact_signature:
        raise ValueError("预测缓存 artifact_signature 与当前模型产物不匹配")
    if current_signature and current_signature == artifact_signature:
        provenance_status = "verified_embedded_metadata"
    elif embedded_model_id and artifact_signature:
        provenance_status = "embedded_metadata_unverified_artifacts"
    else:
        provenance_status = "unresolved_cache_provenance"
    score_semantics = embedded.get("score_semantics") or "unknown_legacy_cache"
    return frame, {
        "path": str(path),
        "file_name": path.name,
        "sha256": _sha256(path),
        "row_count": len(frame),
        "date_start": frame["date"].min().strftime("%Y-%m-%d"),
        "date_end": frame["date"].max().strftime("%Y-%m-%d"),
        "embedded_keys": sorted(str(key) for key in payload) if isinstance(payload, dict) else [],
        "embedded_metadata": embedded,
        "model_id": embedded_model_id,
        "artifact_signature": artifact_signature,
        "current_artifact_signature": current_signature,
        "score_semantics": score_semantics,
        "raw_model_score_available": score_semantics == "raw_model_score",
        "raw_component_scores_available": bool(valid_components),
        "raw_component_keys": valid_components,
        "provenance_status": provenance_status,
    }


def load_history_pool(path, prefixes=DEFAULT_PREFIXES):
    path = Path(path).resolve()
    intervals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code = norm_code(parts[0])
        if len(code) != 6 or not code.startswith(tuple(prefixes)):
            continue
        try:
            start = pd.Timestamp(parts[1]).normalize()
            end = pd.Timestamp(parts[2]).normalize()
        except (TypeError, ValueError):
            continue
        intervals.setdefault(code, []).append((start, end))
    if not intervals:
        raise ValueError("历史筛选池为空")
    return intervals, {
        "semantic_name": "project_a500_history_pool",
        "not_an_index_constituent_claim": True,
        "path": str(path),
        "sha256": _sha256(path),
        "code_count": len(intervals),
        "prefixes": list(prefixes),
    }


def pool_members_on(intervals, date):
    value = pd.Timestamp(date).normalize()
    return {code for code, ranges in intervals.items()
            if any(start <= value <= end for start, end in ranges)}


def _calendar(start, end):
    rows = Database.fetchall(
        "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
        (str(start)[:10], str(end)[:10]),
    )
    return [pd.Timestamp(row["trade_date"]).normalize() for row in rows]


def _industry_map():
    raw = DataFetcher().get_industry_map() or {}
    return {norm_code(code): value for code, value in raw.items()}


def _score_maps(frame, dates, intervals):
    result = {}
    for date in dates:
        members = pool_members_on(intervals, date)
        group = frame.loc[frame["date"] == date, ["code", "score"]]
        values = {}
        for row in group.itertuples(index=False):
            if row.code in members and math.isfinite(float(row.score)):
                values[row.code] = float(row.score)
        if values:
            result[date.strftime("%Y-%m-%d")] = values
    return result


def _latest_date():
    row = Database.fetchone("SELECT MAX(trade_date) AS latest FROM daily_quotes")
    if not row or not row.get("latest"):
        raise RuntimeError("daily_quotes 没有行情")
    return pd.Timestamp(row["latest"]).normalize()


def _load_price_cache(codes, start, end):
    cache = DailyQuotesRepo.get_range_batch(sorted(codes), str(start)[:10], str(end)[:10])
    for code, frame in cache.items():
        cache[code] = frame.sort_values("trade_date").reset_index(drop=True)
    return cache


def _price_indexes(price_cache):
    result = {}
    for code, frame in price_cache.items():
        result[code] = {
            pd.Timestamp(value).strftime("%Y-%m-%d"): index
            for index, value in enumerate(frame["trade_date"])
        }
    return result


def _price_cache_fingerprint(price_cache):
    digest = hashlib.sha256()
    for code in sorted(price_cache):
        frame = price_cache[code]
        columns = [name for name in (
            "trade_date", "open", "close", "volume", "adj_factor") if name in frame.columns]
        digest.update(code.encode("ascii", errors="ignore"))
        digest.update(pd.util.hash_pandas_object(
            frame[columns].reset_index(drop=True), index=False).values.tobytes())
    return digest.hexdigest()


def _number(row, field):
    try:
        value = float(row.get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _execution_outcomes(raw_by_date, dates, calendar, price_cache, price_indexes,
                        industry_map, horizon, max_exit_delay):
    calendar_positions = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(calendar)}
    returns_by_date = {}
    entry_eligible_by_date = {}
    diagnostics = {}
    for date in dates:
        date_key = date.strftime("%Y-%m-%d")
        position = calendar_positions[date_key]
        entry_date = calendar[position + 1]
        planned_exit = calendar[position + horizon + 1]
        entry_key = entry_date.strftime("%Y-%m-%d")
        values = {}
        eligible = set()
        blocked = {"missing_entry": 0, "entry_suspended": 0, "entry_limit_up": 0,
                   "missing_exit": 0, "delayed_exit": 0}
        delay_total = 0
        for code in raw_by_date[date_key]:
            frame = price_cache.get(code)
            indexes = price_indexes.get(code, {})
            entry_index = indexes.get(entry_key)
            if frame is None or entry_index is None:
                blocked["missing_entry"] += 1
                continue
            entry_row = frame.iloc[entry_index]
            entry_open = _number(entry_row, "open")
            if entry_open <= 0 or _number(entry_row, "volume") <= 0:
                blocked["entry_suspended"] += 1
                continue
            info = industry_map.get(code, {})
            name = info.get("name", "") if isinstance(info, dict) else ""
            is_st = "ST" in str(name).upper()
            if is_open_limit(frame, entry_index, code, is_st, "up"):
                blocked["entry_limit_up"] += 1
                continue
            eligible.add(code)
            exit_open = None
            used_delay = 0
            for delay in range(max_exit_delay + 1):
                exit_position = position + horizon + 1 + delay
                if exit_position >= len(calendar):
                    break
                exit_key = calendar[exit_position].strftime("%Y-%m-%d")
                exit_index = indexes.get(exit_key)
                if exit_index is None:
                    continue
                exit_row = frame.iloc[exit_index]
                candidate = _number(exit_row, "open")
                if candidate <= 0 or _number(exit_row, "volume") <= 0:
                    continue
                if is_open_limit(frame, exit_index, code, is_st, "down"):
                    continue
                exit_open = candidate
                used_delay = delay
                break
            if exit_open is None:
                blocked["missing_exit"] += 1
                continue
            if used_delay:
                blocked["delayed_exit"] += 1
                delay_total += used_delay
            values[code] = exit_open / entry_open - 1
        returns_by_date[date_key] = values
        entry_eligible_by_date[date_key] = eligible
        diagnostics[date_key] = {
            "entry_date": entry_key,
            "planned_exit_date": planned_exit.strftime("%Y-%m-%d"),
            "raw_universe_count": len(raw_by_date[date_key]),
            "entry_eligible_count": len(eligible),
            "realized_return_count": len(values),
            "mean_exit_delay_days": (
                delay_total / blocked["delayed_exit"] if blocked["delayed_exit"] else 0.0
            ),
            **blocked,
        }
    return returns_by_date, entry_eligible_by_date, diagnostics


def _decision_eligibility(date_key, candidate_codes, price_cache, industry_map, listing_dates):
    result = {}
    reasons = {}
    for code in candidate_codes:
        frame = price_cache.get(code)
        info = industry_map.get(code, {})
        name = info.get("name", "") if isinstance(info, dict) else ""
        latest_adj = None
        if frame is not None and not frame.empty and "adj_factor" in frame.columns:
            values = pd.to_numeric(frame["adj_factor"], errors="coerce").dropna()
            latest_adj = float(values.iloc[-1]) if not values.empty else None
        check = evaluate_decision_day_filter(
            code, frame, date_key, name=name, latest_adj_factor=latest_adj,
            limit_pct=limit_pct_for_code(code, is_st="ST" in str(name).upper()),
            listing_date=listing_dates.get(code),
        )
        reason = check.get("reason")
        result[code] = reason is None
        reasons[reason or "eligible"] = reasons.get(reason or "eligible", 0) + 1
    return result, reasons


def _rank_ic(scores, returns):
    common = sorted(set(scores) & set(returns))
    if len(common) < 30:
        return None
    left = pd.Series([scores[code] for code in common]).rank(method="average")
    right = pd.Series([returns[code] for code in common]).rank(method="average")
    value = left.corr(right)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _assess_variants(replay, provenance_status, strict_pit):
    baseline = replay["variants"]["baseline"]
    result = {}
    for name, value in replay["variants"].items():
        if name == "baseline":
            result[name] = {"verdict": "reference", "production_change_allowed": False}
            continue
        paired = value["paired_vs_baseline"]
        drawdown_delta = value["summary"]["max_drawdown"] - baseline["summary"]["max_drawdown"]
        gates = {
            "positive_mean_delta": paired["mean_net_return_delta"] > 0,
            "majority_positive_affected_periods": (
                paired["conditional_positive_delta_rate"] is not None
                and paired["conditional_positive_delta_rate"] > 0.5
            ),
            "conditional_bootstrap_ci_above_zero": (
                paired["conditional_bootstrap_mean_delta_ci95"][0] is not None
                and paired["conditional_bootstrap_mean_delta_ci95"][0] > 0
            ),
            "early_and_late_positive": (
                paired["early_mean_delta"] is not None and paired["early_mean_delta"] > 0
                and paired["late_mean_delta"] is not None and paired["late_mean_delta"] > 0
            ),
            "drawdown_not_materially_worse": drawdown_delta >= -0.02,
            "turnover_not_materially_worse": paired["mean_turnover_delta"] <= 0.10,
            "complete_selected_labels": value["summary"]["missing_selected_returns"] == 0,
            "resolved_model_provenance": provenance_status == "verified_embedded_metadata",
            "strict_point_in_time_reference_data": bool(strict_pit),
            "multiple_independent_windows": False,
            "full_engine_confirmed": False,
        }
        if not strict_pit:
            verdict = "exploratory_only_non_pit"
        elif not gates["positive_mean_delta"] or not gates["early_and_late_positive"]:
            verdict = "reject_in_this_window"
        elif all(gates[key] for key in (
                "majority_positive_affected_periods", "conditional_bootstrap_ci_above_zero",
                "drawdown_not_materially_worse", "turnover_not_materially_worse",
                "complete_selected_labels", "resolved_model_provenance",
                "strict_point_in_time_reference_data")):
            verdict = "candidate_requires_independent_window_and_full_engine"
        else:
            verdict = "insufficient_evidence"
        result[name] = {
            "verdict": verdict,
            "production_change_allowed": False,
            "gates": gates,
            "drawdown_delta": drawdown_delta,
        }
    return result


@qlib_read_locked
def run_score_ablation(prediction_file, start=None, end=None, model_id=None, top_n=50,
                       horizon=10, rebalance_step=10, max_dates=None, candidate_depth=500,
                       max_exit_delay=5, output_dir=None, allow_non_pit_reference=False):
    from core.adapters.qlib.generation import assert_no_active_historical_release

    assert_no_active_historical_release(config.QLIB_DATA_ROOT)
    if int(top_n) <= 0 or int(horizon) <= 0 or int(rebalance_step) <= 0:
        raise ValueError("top_n、horizon、rebalance_step 必须为正数")
    if int(candidate_depth) < int(top_n):
        raise ValueError("candidate_depth 不能小于 top_n")
    if int(max_exit_delay) < 0:
        raise ValueError("max_exit_delay 不能为负数")
    if not Database.is_available():
        raise RuntimeError("MySQL 不可用")
    prediction, prediction_meta = load_prediction_cache(
        prediction_file, expected_model_id=model_id)
    if not allow_non_pit_reference:
        raise RuntimeError(
            "项目暂无有效日期化的历史行业和证券名称快照，"
            "严格 PIT P2 已阻止；仅限探索时显式设置 allow_non_pit_reference=True")
    pool_path = Path(config.QLIB_DATA_DIR) / "instruments" / "a500.txt"
    intervals, pool_meta = load_history_pool(pool_path)
    latest = _latest_date()
    requested_start = pd.Timestamp(start or prediction_meta["date_start"]).normalize()
    requested_end = min(pd.Timestamp(end or prediction_meta["date_end"]).normalize(), latest)
    calendar = _calendar(requested_start, latest)
    if not calendar:
        raise RuntimeError("请求区间没有交易日")
    prediction_dates = set(prediction["date"].unique())
    dates = []
    for index in range(0, len(calendar), int(rebalance_step)):
        date = calendar[index]
        if date > requested_end:
            break
        if date not in prediction_dates or index + int(horizon) + 1 + int(max_exit_delay) >= len(calendar):
            continue
        dates.append(date)
        if max_dates and len(dates) >= int(max_dates):
            break
    if not dates:
        raise RuntimeError("没有同时满足预测、成交和标签成熟条件的决策日")
    raw_by_date = _score_maps(prediction, dates, intervals)
    dates = [date for date in dates if date.strftime("%Y-%m-%d") in raw_by_date]
    industry_map = _industry_map()
    input_semantics = prediction_meta["score_semantics"]
    full_scores = {name: {} for name in (
        "baseline", "no_neutralize", "input_rank_control", "macro_bias", "no_decision_filter"
    )}
    processing_audits = {}
    macro_diagnostics = {}
    for date in dates:
        date_key = date.strftime("%Y-%m-%d")
        raw = raw_by_date[date_key]
        neutralized = normalize_qlib_scores(
            raw, date, industry_map=industry_map, neutralize=True, return_details=True,
            model_input_semantics=input_semantics,
        )
        percentile = normalize_qlib_scores(
            raw, date, industry_map=industry_map, neutralize=False, return_details=True,
            model_input_semantics=input_semantics,
        )
        macro_scores, macro_signals = apply_macro_sector_bias(
            dict(neutralized["scores"]), date, industry_map=industry_map
        )
        full_scores["baseline"][date_key] = neutralized["scores"]
        full_scores["no_neutralize"][date_key] = percentile["scores"]
        full_scores["input_rank_control"][date_key] = raw
        full_scores["macro_bias"][date_key] = macro_scores
        full_scores["no_decision_filter"][date_key] = neutralized["scores"]
        processing_audits[date_key] = {
            "source_fingerprint": score_fingerprint(raw),
            "neutralized": neutralized["audit"],
            "percentile_only": percentile["audit"],
        }
        macro_diagnostics[date_key] = {
            "signals": macro_signals,
            "changed_score_count": sum(
                abs(macro_scores.get(code, 0) - neutralized["scores"].get(code, 0)) > 1e-12
                for code in neutralized["scores"]
            ),
        }
    all_codes = set().union(*(set(values) for values in raw_by_date.values()))
    try:
        from core.data import StockIdentityRepo
        listing_dates = StockIdentityRepo.get_listing_dates(all_codes)
    except Exception as error:
        logger.warning("[ScoreAblation] 证券上市日期不可用，相关候选将按未知排除: %s", error)
        listing_dates = {}
    price_start = dates[0] - timedelta(days=200)
    last_position = {value.strftime("%Y-%m-%d"): index for index, value in enumerate(calendar)}[
        dates[-1].strftime("%Y-%m-%d")]
    price_end = calendar[last_position + int(horizon) + 1 + int(max_exit_delay)]
    price_cache = _load_price_cache(all_codes, price_start, price_end)
    price_snapshot_fingerprint = _price_cache_fingerprint(price_cache)
    price_indexes = _price_indexes(price_cache)
    returns_by_date, entry_eligible, execution_diagnostics = _execution_outcomes(
        raw_by_date, dates, calendar, price_cache, price_indexes,
        industry_map, int(horizon), int(max_exit_delay),
    )
    filtered_scores = {name: {} for name in full_scores}
    decision_diagnostics = {}
    filter_variants = {"baseline", "no_neutralize", "input_rank_control", "macro_bias"}
    for date in dates:
        date_key = date.strftime("%Y-%m-%d")
        candidates = set()
        ranked_by_variant = {}
        for name, by_date in full_scores.items():
            ranked = sorted(by_date[date_key], key=lambda code: (-by_date[date_key][code], code))
            ranked_by_variant[name] = ranked[:int(candidate_depth)]
            candidates.update(ranked_by_variant[name])
        decision_ok, reasons = _decision_eligibility(
            date_key, candidates, price_cache, industry_map, listing_dates
        )
        decision_diagnostics[date_key] = {
            "evaluated_count": len(candidates),
            "reason_counts": reasons,
        }
        for name in full_scores:
            allowed = set(ranked_by_variant[name])
            if name in filter_variants:
                allowed = {code for code in allowed if decision_ok.get(code, False)}
            filtered_scores[name][date_key] = {
                code: full_scores[name][date_key][code]
                for code in ranked_by_variant[name] if code in allowed
            }
    variants = {
        name: {
            "scores_by_date": values,
            "config": {
                "neutralize": name not in ("no_neutralize", "input_rank_control"),
                "percentile_mapping": name != "input_rank_control",
                "macro_bias": name == "macro_bias",
                "decision_day_filter": name != "no_decision_filter",
                "entry_execution_policy": "reject_without_refill_and_hold_cash",
            },
        }
        for name, values in filtered_scores.items()
    }
    replay = replay_score_variants(
        raw_by_date, variants, int(top_n), returns_by_date,
        baseline="baseline", buy_cost_rate=0.0004, sell_cost_rate=0.0009,
        missing_return_policy="cash",
        entry_eligible_by_date=entry_eligible,
    )
    invariant = []
    for left, right in zip(
            replay["variants"]["no_neutralize"]["periods"],
            replay["variants"]["input_rank_control"]["periods"]):
        invariant.append(left["selected"] == right["selected"]
                         and abs(left["net_return"] - right["net_return"]) < 1e-12)
    if not invariant or not all(invariant):
        raise RuntimeError("分位映射控制组改变了排序或收益，实验契约失败")
    rank_ic = {}
    for name, by_date in full_scores.items():
        values = [_rank_ic(by_date[date_key], returns_by_date[date_key])
                  for date_key in sorted(raw_by_date)]
        clean = [value for value in values if value is not None]
        rank_ic[name] = {
            "periods": values,
            "mean": sum(clean) / len(clean) if clean else None,
            "positive_rate": sum(value > 0 for value in clean) / len(clean) if clean else None,
            "valid_periods": len(clean),
        }
    config_payload = {
        "label_mode": "tradable_open_to_open",
        "decision_to_entry": "T+1_open",
        "horizon": int(horizon),
        "rebalance_step": int(rebalance_step),
        "top_n": int(top_n),
        "candidate_depth": int(candidate_depth),
        "max_exit_delay": int(max_exit_delay),
        "costs": {"buy": 0.0004, "sell": 0.0009},
        "start": dates[0].strftime("%Y-%m-%d"),
        "end": dates[-1].strftime("%Y-%m-%d"),
        "model_id": prediction_meta.get("model_id"),
        "expected_model_id": model_id,
        "input_score_semantics": input_semantics,
        "daily_quotes_latest_date": latest.strftime("%Y-%m-%d"),
        "implementation_hashes": _implementation_hashes(),
    }
    data_snapshot = {
        "prediction_sha256": prediction_meta["sha256"],
        "pool_sha256": pool_meta["sha256"],
        "price_rows_fingerprint": price_snapshot_fingerprint,
        "industry_name_snapshot_fingerprint": _stable_hash(industry_map),
        "processing_input_fingerprint": _stable_hash(processing_audits),
        "macro_signal_fingerprint": _stable_hash(macro_diagnostics),
        "execution_outcome_fingerprint": _stable_hash({
            "returns": returns_by_date,
            "diagnostics": execution_diagnostics,
            "entry_eligible": entry_eligible,
        }),
        "decision_filter_fingerprint": _stable_hash(decision_diagnostics),
    }
    data_snapshot["fingerprint"] = _stable_hash(data_snapshot)
    report = {
        "schema_version": 2,
        "experiment": "score_ablation_p2",
        "experiment_fingerprint": _stable_hash({
            "config": config_payload,
            "data_snapshot_fingerprint": data_snapshot["fingerprint"],
        }),
        "config": config_payload,
        "prediction_source": {
            **prediction_meta,
        },
        "pool": pool_meta,
        "data_snapshot": data_snapshot,
        "decision_dates": [date.strftime("%Y-%m-%d") for date in dates],
        "replay": replay,
        "rank_ic": rank_ic,
        "execution_diagnostics": execution_diagnostics,
        "decision_filter_diagnostics": decision_diagnostics,
        "macro_diagnostics": macro_diagnostics,
        "processing_audits": processing_audits,
        "controls": {
            "input_vs_percentile_selection_invariant": all(invariant),
            "input_vs_percentile_periods_checked": len(invariant),
            "same_source_fingerprint_per_date": True,
            "cross_date_quality_evaluable": input_semantics == "raw_model_score",
            "input_score_semantics": input_semantics,
        },
        "unsupported_variants": {
            "sector_fusion": "requires direction-specific multi-model prediction bundle; single-cache replay would be invalid",
        },
        "evidence_assessment": _assess_variants(
            replay, prediction_meta["provenance_status"], strict_pit=False),
        "limitations": [
            "score-level paired replay is an exploratory estimate, not a full position-locking engine backtest",
            "explicit non-PIT override uses current security name and industry snapshots",
            "delayed exits can overlap the next rebalance and require full-engine confirmation",
            "a single cache or window cannot authorize a production default change",
            "sector fusion is not evaluated without direction-specific prediction caches",
        ],
        "production_change_allowed": False,
        "evidence_class": "exploratory_non_pit",
    }
    base = Path(output_dir) if output_dir else (
        Path("data") / "experiments" / "score_ablation_p2"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    base.mkdir(parents=True, exist_ok=False)
    ledger_path = base / "score_ledger.pkl.gz"
    with gzip.open(ledger_path, "wb") as handle:
        pickle.dump({
            "raw_by_date": raw_by_date,
            "full_scores": full_scores,
            "filtered_scores": filtered_scores,
            "returns_by_date": returns_by_date,
            "entry_eligible_by_date": entry_eligible,
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report["artifacts"] = {
        "score_ledger": str(ledger_path.resolve()),
        "score_ledger_sha256": _sha256(ledger_path),
    }
    report_path = base / "report.json"
    report_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report, report_path.resolve()
