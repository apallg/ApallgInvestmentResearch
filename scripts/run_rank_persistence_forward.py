import argparse
import hashlib
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.adapters.qlib.model_contract import get_model_label_mode
from core.data import Database
from core.domain.backtest.engine import DirectionBacktestRunner
from core.domain.backtest import engine as backtest_engine
from core.domain.backtest.common import norm_code
from core.shared import config
from core.shared.trading_time import get_latest_trading_day, get_trading_days_between, is_trading_day
from scripts.analyze_rank_persistence_ab import summarize_delta
from scripts.capture_rank_persistence_shadow import (
    CONFIRM_SESSIONS,
    CONTRACT_FILES,
    MODEL,
    OUTPUT,
    POOL,
    PORTFOLIO_CODE_PREFIXES,
    REQUIRED_APPEARANCES,
    TOP_DEPTH,
    contract_fingerprint_for,
    file_sha256,
    validate_prior_snapshot,
)
from scripts.run_rank_persistence_ab import (
    align_period_returns_to_decisions,
    fast_precompute_limit_down,
    fast_precompute_limit_up,
)


TOP_N = 5
PRIMARY_PILOT_PERIODS = 52


def model_fingerprint_for_run(run_path):
    artifact_paths = [
        run_path / "artifacts" / "params.pkl",
        run_path / "artifacts" / "params_lstm.pkl",
        run_path / "artifacts" / "lstm_bundle.pkl",
        run_path / "artifacts" / "factor_keys.json",
        run_path / "artifacts" / "pred.pkl",
    ]
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"固定模型文件不完整: {missing}")
    fingerprint = hashlib.sha256("".join(
        f"{path.name}:{file_sha256(path)}\n" for path in artifact_paths
    ).encode("utf-8")).hexdigest()
    return fingerprint


def prediction_from_snapshots(snapshots):
    keys = []
    values = []
    for snapshot in snapshots:
        prediction_date = snapshot["prediction_date"]
        for row in snapshot["ranked_scores"]:
            keys.append((pd.Timestamp(prediction_date), str(row["code"])))
            values.append(float(row["score"]))
    index = pd.MultiIndex.from_tuples(keys, names=["datetime", "instrument"])
    prediction = pd.Series(values, index=index, name="score").sort_index()
    if prediction.index.has_duplicates:
        raise RuntimeError("前瞻快照包含重复日期、股票键")
    return prediction


def history_from_snapshots(snapshots):
    prediction = prediction_from_snapshots(snapshots)
    rank_by_date = {}
    for snapshot in snapshots:
        rows = snapshot["ranked_scores"]
        rank_by_date[snapshot["prediction_date"]] = {
            str(row["code"]): int(row["rank"]) for row in rows
        }
    dates = sorted(rank_by_date)
    stable_by_date = {}
    count_by_date = {}
    for index, prediction_date in enumerate(dates):
        current = rank_by_date[prediction_date]
        counts = {
            code: sum(
                rank_by_date[dates[position]].get(code, TOP_DEPTH + 1) <= TOP_DEPTH
                for position in range(max(0, index - CONFIRM_SESSIONS + 1), index + 1)
            )
            for code in current
        }
        stable_by_date[prediction_date] = {
            code for code, rank in current.items()
            if rank <= TOP_DEPTH and counts[code] >= REQUIRED_APPEARANCES
        }
        count_by_date[prediction_date] = counts
    return prediction, rank_by_date, stable_by_date, count_by_date


def qlib_calendar_through(as_of):
    qlib_dir = Path(config.QLIB_DATA_DIR)
    calendar_path = qlib_dir / "calendars" / "day.txt"
    calendar = [date.fromisoformat(value) for value in
                calendar_path.read_text(encoding="utf-8").splitlines()]
    if not calendar or calendar[-1] != as_of:
        raise RuntimeError("Qlib日历末日与daily_quotes不一致，拒绝评估")
    return [value for value in calendar if value <= as_of]


def load_contract_snapshots(as_of, model_fingerprint, pool_fingerprint,
                            contract_fingerprint, calendar_dates):
    latest = Database.fetchone("SELECT MAX(trade_date) AS latest FROM daily_quotes")
    latest_quote_date = latest.get("latest") if latest else None
    if hasattr(latest_quote_date, "date"):
        latest_quote_date = latest_quote_date.date()
    if latest_quote_date != as_of:
        raise RuntimeError(
            f"回放截止日必须等于daily_quotes最新日期: latest={latest_quote_date}, requested={as_of}"
        )
    if as_of != get_latest_trading_day(date.today()):
        raise RuntimeError("daily_quotes最新日期与项目交易日历不一致，拒绝评估")

    raw = {}
    for path in sorted(OUTPUT.glob("snapshot_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        prediction_date = snapshot.get("prediction_date")
        if not prediction_date or date.fromisoformat(prediction_date) > as_of:
            continue
        if int(snapshot.get("schema_version", 1)) < 3:
            continue
        if not validate_prior_snapshot(
                snapshot, prediction_date, model_fingerprint, pool_fingerprint,
                contract_fingerprint):
            raise RuntimeError(f"新版本快照不具备可复现合同: {path}")
        raw[prediction_date] = snapshot
    if as_of.isoformat() not in raw:
        return [], calendar_dates

    if as_of not in calendar_dates:
        raise RuntimeError("Qlib日历不包含回放截止日")
    consecutive = []
    for trading_date in reversed(calendar_dates):
        key = trading_date.isoformat()
        if key not in raw:
            break
        consecutive.append(raw[key])
    snapshots = list(reversed(consecutive))
    if not snapshots:
        raise RuntimeError("找不到连续有效的日度快照")
    return snapshots, calendar_dates


def current_weekly_anchor(as_of, calendar_dates, today=None):
    today = today or date.today()
    weekday = as_of.weekday()
    thursday = as_of + timedelta(days=(3 - weekday) % 7)
    if thursday < as_of:
        return None
    if thursday == as_of:
        return as_of
    if today != thursday or is_trading_day(datetime.combine(thursday, time(12, 0))):
        return None
    sessions = get_trading_days_between(thursday - timedelta(days=7), thursday + timedelta(days=1))
    sessions = [value for value in sessions if value <= thursday and value in calendar_dates]
    return max(sessions) if sessions else None


def terminal_execution_session(decision_end, calendar_dates):
    return next((value for value in calendar_dates if value > decision_end), None)


def weekly_decision_dates(calendar_dates, snapshot_dates, through, current_anchor=None):
    calendar_set = set(calendar_dates)
    snapshot_set = set(snapshot_dates)
    thursdays = sorted({
        day + timedelta(days=(3 - day.weekday()) % 7)
        for day in calendar_dates if day <= through
    })
    decisions = []
    for thursday in thursdays:
        if thursday > through:
            if (current_anchor is not None and current_anchor <= through
                    and current_anchor.weekday() < 3
                    and current_anchor + timedelta(days=(3 - current_anchor.weekday())) == thursday
                    and current_anchor in snapshot_set):
                decisions.append(current_anchor)
            continue
        if thursday in calendar_set:
            decision = thursday
        else:
            prior_sessions = [
            day for day in calendar_dates
                if thursday - timedelta(days=7) < day < thursday
            ]
            decision = max(prior_sessions) if prior_sessions else None
        if decision and decision in snapshot_set:
            decisions.append(decision)
    return sorted(set(decisions))


def pre_registered_primary_window(eligible_dates):
    if len(eligible_dates) < PRIMARY_PILOT_PERIODS + 1:
        return None
    primary_dates = eligible_dates[:PRIMARY_PILOT_PERIODS + 1]
    return primary_dates[0], primary_dates[-1], primary_dates


def build_audit_rows(variant, selections, rank_by_date, stable_by_date, count_by_date,
                     returns_by_decision):
    rows = []
    for decision_date, codes in selections:
        outcome = returns_by_decision.get(decision_date, {})
        for slot, code in enumerate(codes[:TOP_N], 1):
            normalized = norm_code(code)
            rows.append({
                "variant": variant,
                "decision_date": decision_date,
                "slot": slot,
                "code": normalized,
                "same_day_rank": rank_by_date.get(decision_date, {}).get(normalized),
                "appearances_in_window": count_by_date.get(decision_date, {}).get(normalized, 0),
                "confirmed": normalized in stable_by_date.get(decision_date, set()),
                "period_return": outcome.get("period_return"),
                "return_period_end_date": outcome.get("outcome_end_date"),
            })
    return rows


def dataframe_fingerprint(frame):
    if frame is None or frame.empty:
        raise RuntimeError("用于评估的数据表为空，拒绝生成指纹")
    ordered = frame.sort_values(
        [column for column in ("trade_date", "date", "code") if column in frame.columns]
    ) if any(column in frame.columns for column in ("trade_date", "date", "code")) else frame
    digest = hashlib.sha256()
    columns = list(ordered.columns)
    for values in ordered.itertuples(index=False, name=None):
        row = []
        for value in values:
            if pd.isna(value):
                row.append(None)
            elif hasattr(value, "isoformat"):
                row.append(value.isoformat())
            else:
                row.append(str(value))
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "row_count": len(ordered), "columns": columns}


def price_cache_fingerprint(price_cache):
    digest = hashlib.sha256()
    row_count = 0
    missing_adjustment_rows = 0
    codes_with_rows = 0
    for code in sorted(price_cache):
        frame = price_cache[code]
        if frame is None or frame.empty:
            continue
        codes_with_rows += 1
        ordered = frame.sort_values("trade_date") if "trade_date" in frame.columns else frame
        row_count += len(ordered)
        if "adj_factor" in ordered.columns:
            missing_adjustment_rows += int(pd.to_numeric(
                ordered["adj_factor"], errors="coerce"
            ).isna().sum())
        digest.update(str(code).encode("utf-8"))
        digest.update(b"\n")
        digest.update(dataframe_fingerprint(ordered)["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "codes_with_rows": codes_with_rows,
        "code_count_in_cache": len(price_cache),
        "price_rows": row_count,
        "missing_adj_factor_rows": missing_adjustment_rows,
    }


def json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"评估产物含不可序列化类型: {type(value).__name__}")


def canonical_report_hash(report):
    serialized = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=json_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_existing_evaluation(report_path, audit_path, evaluation_id):
    if not report_path.is_file() or not audit_path.is_file():
        raise RuntimeError("前瞻评估产物不完整，拒绝将半份文件视为已完成")
    existing = json.loads(report_path.read_text(encoding="utf-8"))
    if existing.get("evaluation_id") != evaluation_id:
        raise RuntimeError(f"评估日期产物已存在但输入清单不同，拒绝覆盖: {report_path}")
    report_hash = existing.pop("report_sha256", None)
    if not report_hash or canonical_report_hash(existing) != report_hash:
        raise RuntimeError(f"评估报告内容哈希校验失败: {report_path}")
    audit_hash = existing.get("audit_sha256")
    if not audit_hash or hashlib.sha256(audit_path.read_bytes()).hexdigest() != audit_hash:
        raise RuntimeError(f"逐股审计表哈希校验失败: {audit_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", help="默认使用daily_quotes与Qlib共同确认的最新交易日")
    args = parser.parse_args()

    latest_row = Database.fetchone("SELECT MAX(trade_date) AS latest FROM daily_quotes")
    if not latest_row or not latest_row.get("latest"):
        raise RuntimeError("daily_quotes没有最新交易日")
    latest_quote_date = latest_row["latest"]
    if hasattr(latest_quote_date, "date"):
        latest_quote_date = latest_quote_date.date()
    as_of = date.fromisoformat(args.as_of) if args.as_of else latest_quote_date
    if args.as_of and as_of != date.today() and as_of != get_latest_trading_day(date.today()):
        raise RuntimeError("仅允许在项目最新交易日收盘后评估")
    if as_of != latest_quote_date:
        raise RuntimeError("评估日期不是daily_quotes最新日期")
    calendar_dates = qlib_calendar_through(as_of)
    anchor = current_weekly_anchor(as_of, calendar_dates)
    if anchor is None or anchor != as_of:
        print(json.dumps({
            "status": "not_weekly_decision_date",
            "as_of": as_of.isoformat(),
        }, ensure_ascii=False))
        return
    latest_snapshot_path = OUTPUT / f"snapshot_{as_of.strftime('%Y%m%d')}.json"
    if not latest_snapshot_path.is_file():
        print(json.dumps({
            "status": "not_ready",
            "reason": "latest_trading_date_has_no_snapshot",
            "as_of": as_of.isoformat(),
        }, ensure_ascii=False))
        return
    latest_snapshot_header = json.loads(latest_snapshot_path.read_text(encoding="utf-8"))
    if int(latest_snapshot_header.get("schema_version", 1)) < 3:
        print(json.dumps({
            "status": "not_ready",
            "reason": "latest_snapshot_is_legacy",
            "as_of": as_of.isoformat(),
        }, ensure_ascii=False))
        return
    evaluation_dir = OUTPUT / "evaluations"
    primary_report_path = evaluation_dir / "primary_52w.json"
    primary_audit_path = evaluation_dir / "primary_52w_stocks.csv"
    if primary_report_path.exists() or primary_audit_path.exists():
        existing = json.loads(primary_report_path.read_text(encoding="utf-8")) if primary_report_path.exists() else {}
        validate_existing_evaluation(
            primary_report_path, primary_audit_path, existing.get("evaluation_id"),
        )
        print(json.dumps({
            "status": "already_evaluated",
            "primary_endpoint": existing.get("primary_endpoint"),
            "path": str(primary_report_path),
        }, ensure_ascii=False))
        return

    mlruns = Path(str(config.QLIB_MLFLOW_URI).replace("file:", ""))
    run_path = mlruns / MODEL[0] / MODEL[1]
    if get_model_label_mode(run_path) != "tradable_open_to_open":
        raise RuntimeError("固定模型标签合同不匹配")
    model_fingerprint = model_fingerprint_for_run(run_path)
    pool_path = Path(config.QLIB_DATA_DIR) / "instruments" / "a500.txt"
    if not pool_path.is_file():
        raise RuntimeError(f"模型股票池文件不存在: {pool_path}")
    pool_fingerprint = file_sha256(pool_path)
    contract_files_sha256 = {
        relative_path: file_sha256(ROOT / relative_path)
        for relative_path in CONTRACT_FILES
    }
    contract_fingerprint = contract_fingerprint_for(contract_files_sha256)
    snapshots, calendar_dates = load_contract_snapshots(
        as_of, model_fingerprint, pool_fingerprint, contract_fingerprint,
        calendar_dates,
    )
    if not snapshots:
        print(json.dumps({
            "status": "not_ready",
            "reason": "latest_trading_date_has_no_valid_schema_v3_snapshot",
            "as_of": as_of.isoformat(),
        }, ensure_ascii=False))
        return

    latest_snapshot_date = date.fromisoformat(snapshots[-1]["prediction_date"])
    anchor = current_weekly_anchor(latest_snapshot_date, calendar_dates)
    if anchor is None or anchor != latest_snapshot_date:
        print(json.dumps({
            "status": "not_weekly_decision_date",
            "latest_snapshot_date": latest_snapshot_date.isoformat(),
            "as_of": as_of.isoformat(),
        }, ensure_ascii=False))
        return

    prediction, rank_by_date, stable_by_date, count_by_date = history_from_snapshots(snapshots)
    snapshot_dates = {row["prediction_date"] for row in snapshots}
    weekly_dates = weekly_decision_dates(
        calendar_dates, [date.fromisoformat(value) for value in snapshot_dates],
        as_of, current_anchor=anchor,
    )
    prediction_dates = [date.fromisoformat(value) for value in rank_by_date]
    eligible_dates = [
        value for value in weekly_dates
        if sum(day <= value for day in prediction_dates) >= CONFIRM_SESSIONS
    ]
    primary_window = pre_registered_primary_window(eligible_dates)
    if primary_window is None:
        print(json.dumps({
            "status": "not_ready",
            "valid_consecutive_daily_snapshots": len(snapshots),
            "weekly_decision_dates": [value.isoformat() for value in eligible_dates],
            "required_matured_periods_for_single_primary_look": PRIMARY_PILOT_PERIODS,
        }, ensure_ascii=False))
        return

    primary_start, primary_endpoint, primary_dates = primary_window
    primary_endpoint_key = primary_endpoint.isoformat()
    terminal_execution_date = terminal_execution_session(primary_endpoint, calendar_dates)
    if terminal_execution_date is None:
        print(json.dumps({
            "status": "not_ready",
            "reason": "terminal_execution_session_not_matured",
            "decision_end": primary_endpoint_key,
            "as_of": as_of.isoformat(),
        }, ensure_ascii=False))
        return
    snapshots = [row for row in snapshots if row["prediction_date"] <= primary_endpoint_key]
    prediction, rank_by_date, stable_by_date, count_by_date = history_from_snapshots(snapshots)
    snapshot_dates = {row["prediction_date"] for row in snapshots}
    start = (primary_start - timedelta(days=primary_start.weekday())).isoformat()
    end = terminal_execution_date.isoformat()
    references_by_date = {
        row["prediction_date"]: row["security_references"] for row in snapshots
    }
    if any(value.isoformat() not in references_by_date for value in primary_dates):
        raise RuntimeError("有效周度决策日缺少冻结证券参考映射")

    quote_rows = Database.fetchone(
        "SELECT COUNT(*) AS n FROM daily_quotes WHERE trade_date=%s", (as_of,)
    )
    if not quote_rows or int(quote_rows.get("n") or 0) < 1000:
        raise RuntimeError("回放截止日行情行数不足，拒绝生成结果")
    execution_quote_rows = Database.fetchone(
        "SELECT COUNT(*) AS n FROM daily_quotes WHERE trade_date=%s",
        (terminal_execution_date,),
    )
    if not execution_quote_rows or int(execution_quote_rows.get("n") or 0) < 1000:
        raise RuntimeError("终点T+1执行日行情覆盖不足，拒绝生成成熟收益")

    from core.data.fetcher import DataFetcher
    from core.shared.infrastructure.cache import CacheManager
    DataFetcher.get_industry_map = lambda self: {}
    backtest_engine.precompute_limit_up = fast_precompute_limit_up
    backtest_engine.precompute_limit_down = fast_precompute_limit_down
    original_get_redis = CacheManager.get_redis
    original_set_redis = CacheManager.set_redis
    CacheManager.get_redis = lambda self, key: None
    CacheManager.set_redis = lambda self, key, value, expire=3600: False
    runner = DirectionBacktestRunner()
    model_ref = MODEL
    cache_key = (
        start, end, primary_endpoint_key, POOL, ("trend_momentum",),
        tuple(PORTFOLIO_CODE_PREFIXES), (), model_ref,
    )
    runner._cached_run_key = cache_key
    runner._cached_pred_cache = {"fixed": prediction}
    original_filter = runner._filter_decision_day_ranking
    variants = {}
    audit_rows = []
    selected_dates = []
    for variant in ("A_baseline", "B_stability_first", "C_strict_confirm"):
        runner._cached_pred_cache = {"fixed": prediction}
        initial_selection = {}
        original_initial_build = runner._initial_build

        def capture_initial_build(codes, scores, top_n, price_cache, date_str,
                                  sector_map, next_str=None):
            initial_selection["date"] = str(date_str)[:10]
            initial_selection["codes"] = list(codes[:top_n])
            return original_initial_build(
                codes, scores, top_n, price_cache, date_str, sector_map, next_str
            )

        runner._initial_build = capture_initial_build

        def filtered(ranked_codes, decision_date, target_count, price_cache, sector_map,
                     min_history_observations=None, mode=variant):
            requested_count = target_count if mode == "A_baseline" else max(target_count, 50)
            eligible, excluded, examined = original_filter(
                ranked_codes, decision_date, requested_count, price_cache, sector_map,
                min_history_observations,
            )
            key = str(decision_date)[:10]
            stable = stable_by_date.get(key, set())
            persistent = [code for code in eligible if norm_code(code) in stable]
            if mode == "A_baseline":
                return eligible, excluded, examined
            if mode == "C_strict_confirm":
                return persistent, excluded, examined
            persistent_set = set(persistent)
            return persistent + [code for code in eligible if code not in persistent_set], excluded, examined

        runner._filter_decision_day_ranking = filtered
        result = runner.run(
            start=start,
            end=end,
            pool=POOL,
            top_n=TOP_N,
            freq="week",
            sizing="equal",
            target_exposure=0.95,
            max_one_way_turnover=0.30,
            directions=["trend_momentum"],
            full_position=True,
            code_prefixes=PORTFOLIO_CODE_PREFIXES,
            fixed_model=model_ref,
            commission_rate=0.0003,
            slippage_rate=0.0001,
            stamp_duty=0.0005,
            stop_source="off",
            macro_bias=False,
            sector_fusion=False,
            neutralize_qlib=False,
            decision_day_filter=True,
            security_references_by_date=references_by_date,
            decision_end=primary_endpoint_key,
        )
        runner._initial_build = original_initial_build
        if not initial_selection:
            raise RuntimeError(f"{variant}未生成初始组合")
        returns_by_decision = align_period_returns_to_decisions(
            result.get("equity_curve", []), result.get("periods", [])
        )
        selections = [(initial_selection["date"], initial_selection["codes"])]
        selections.extend(
            (str(period.get("date"))[:10], period.get("top_stocks", [])[:TOP_N])
            for period in result.get("periods", [])
        )
        if variant == "A_baseline":
            selected_dates = [key for key in returns_by_decision]
        elif list(returns_by_decision) != selected_dates:
            raise RuntimeError("A/B/C成熟收益日期不一致，拒绝配对比较")
        audit_rows.extend(build_audit_rows(
            variant, selections, rank_by_date, stable_by_date, count_by_date,
            returns_by_decision,
        ))
        variants[variant] = {
            "summary": result.get("summary"),
            "periods": result.get("periods"),
            "equity_curve": result.get("equity_curve"),
            "score_processing": result.get("score_processing"),
        }

    CacheManager.get_redis = original_get_redis
    CacheManager.set_redis = original_set_redis
    runner._filter_decision_day_ranking = original_filter
    price_data_fingerprint = price_cache_fingerprint(runner._cached_price_cache)
    expected_price_codes = {
        norm_code(code) for code in prediction.index.get_level_values("instrument").unique()
        if norm_code(code).startswith(tuple(PORTFOLIO_CODE_PREFIXES))
    }
    price_data_fingerprint["expected_candidate_codes"] = len(expected_price_codes)
    price_data_fingerprint["coverage"] = (
        price_data_fingerprint["codes_with_rows"] / len(expected_price_codes)
        if expected_price_codes else 0.0
    )
    index_data_fingerprint = dataframe_fingerprint(runner._cached_index_df)
    returns = {
        variant: align_period_returns_to_decisions(
            values["equity_curve"], values["periods"]
        )
        for variant, values in variants.items()
    }
    paired = {}
    for variant in ("B_stability_first", "C_strict_confirm"):
        if set(returns["A_baseline"]) != set(returns[variant]):
            raise RuntimeError(f"{variant}与基准组成熟期不匹配")
        deltas = [
            float(returns[variant][key]["period_return"])
            - float(returns["A_baseline"][key]["period_return"])
            for key in sorted(returns["A_baseline"])
        ]
        paired[f"{variant}_minus_A"] = summarize_delta(
            deltas, block_size=4, draws=10000, seed=20260925,
        )

    b_ci = paired["B_stability_first_minus_A"]["bootstrap_ci95_pp_per_week"]
    matured_periods = len(returns["A_baseline"])
    if matured_periods != PRIMARY_PILOT_PERIODS:
        raise RuntimeError(
            f"预注册试点终点要求{PRIMARY_PILOT_PERIODS}个成熟周，实际为{matured_periods}，拒绝更改检验窗口"
        )
    pilot_statistical_signal_met = matured_periods == PRIMARY_PILOT_PERIODS and b_ci[0] > 0
    evaluation_manifest = {
        "as_of": as_of.isoformat(),
        "weekly_anchor": primary_endpoint_key,
        "decision_end": primary_endpoint_key,
        "market_data_end": end,
        "primary_endpoint": {
            "decision_date": primary_endpoint_key,
            "matured_paired_periods": PRIMARY_PILOT_PERIODS,
            "single_look": True,
        },
        "snapshot_dates": [row["prediction_date"] for row in snapshots],
        "snapshot_score_hashes": [row["scores_sha256"] for row in snapshots],
        "snapshot_security_hashes": [row["security_references_sha256"] for row in snapshots],
        "contract_fingerprint": contract_fingerprint,
        "price_data_fingerprint": price_data_fingerprint["sha256"],
        "index_data_fingerprint": index_data_fingerprint["sha256"],
    }
    evaluation_id = hashlib.sha256(json.dumps(
        evaluation_manifest, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    report = {
        "status": "forward_shadow_primary_evaluation",
        "interpretation": "预注册单次52周前瞻影子检验，不下单；不得在该终点后反复查看或改窗口。",
        "evaluation_id": evaluation_id,
        "manifest": evaluation_manifest,
        "model": {"experiment_id": MODEL[0], "run_id": MODEL[1],
                  "label_mode": "tradable_open_to_open",
                  "fingerprint": model_fingerprint},
        "pool": {"name": POOL, "fingerprint": pool_fingerprint},
        "data_audit": {
            "price_source": "daily_quotes via DirectionBacktestRunner with Redis price-cache reads/writes disabled",
            "price": price_data_fingerprint,
            "index_source": "DataFetcher.get_index_daily('沪深300', days=2000); exact returned frame fingerprint",
            "index": index_data_fingerprint,
        },
        "execution_contract": {
            "start": start, "end": end, "decision_end": primary_endpoint_key,
            "terminal_execution_date": terminal_execution_date.isoformat(),
            "top_n": TOP_N, "frequency": "week",
            "target_exposure": 0.95, "max_one_way_turnover": 0.30,
            "commission_rate": 0.0003, "slippage_rate": 0.0001,
            "stamp_duty": 0.0005, "stop_source": "off",
            "neutralize_qlib": False, "decision_day_filter": True,
            "security_references_by_date": True,
            "confirmation": {"depth": TOP_DEPTH, "sessions": CONFIRM_SESSIONS,
                             "required_appearances": REQUIRED_APPEARANCES},
        },
        "matured_paired_periods": matured_periods,
        "primary_endpoint": primary_endpoint_key,
        "pilot_result": {
            "pre_registered_matured_periods": PRIMARY_PILOT_PERIODS,
            "sample_size_met": matured_periods == PRIMARY_PILOT_PERIODS,
            "B_minus_A_ci_lower_above_zero": b_ci[0] > 0,
            "pilot_statistical_signal_met": pilot_statistical_signal_met,
            "production_promotion_ready": False,
            "remaining_gates": [
                "独立审计点时证券/股票池来源",
                "按当期NAV复核实际双边换手与成本",
                "成交可行性、涨跌停及复权价格口径复核",
                "另一独立预注册前瞻时段确认",
            ],
            "primary_metric": "B-A mean weekly net return; circular 4-week block-bootstrap 95% CI; pilot signal is not production promotion",
        },
        "paired_analysis": paired,
        "variants": variants,
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report_path = primary_report_path
    audit_path = primary_audit_path
    if report_path.exists() or audit_path.exists():
        validate_existing_evaluation(report_path, audit_path, evaluation_id)
        print(json.dumps({"status": "already_evaluated", "path": str(report_path)}, ensure_ascii=False))
        return
    audit_temp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    report_temp = report_path.with_suffix(report_path.suffix + ".tmp")
    if audit_temp.exists() or report_temp.exists():
        raise RuntimeError("发现未完成的评估临时文件，拒绝覆盖并需人工检查")
    audit_text = pd.DataFrame(audit_rows).to_csv(index=False)
    audit_bytes = ("\ufeff" + audit_text).encode("utf-8")
    report["audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    report["report_sha256"] = canonical_report_hash(report)
    report_bytes = json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False, default=json_default,
    ).encode("utf-8")
    audit_temp.write_bytes(audit_bytes)
    report_temp.write_bytes(report_bytes)
    audit_temp.replace(audit_path)
    report_temp.replace(report_path)
    print(json.dumps({
        "status": report["status"],
        "primary_endpoint": primary_endpoint_key,
        "matured_paired_periods": report["matured_paired_periods"],
        "pilot_statistical_signal_met": report["pilot_result"]["pilot_statistical_signal_met"],
        "report": str(report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
