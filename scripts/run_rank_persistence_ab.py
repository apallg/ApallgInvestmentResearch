import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.domain.backtest.engine import DirectionBacktestRunner
from core.domain.backtest.common import norm_code
from core.domain.backtest import common as backtest_common
from core.domain.backtest import engine as backtest_engine
from core.adapters.qlib.model_contract import get_model_label_mode
from core.shared import config
from core.data.fetcher import DataFetcher


MODEL_EXP = "656749665990832865"
MODEL_RUN = "7e5e67ba9b714c76b0808a2626d0f7d1"
START = "2025-07-01"
END = "2026-08-31"
TOP_N = 5
CONFIRM_DEPTH = 20
CONFIRM_WINDOW = 3
CONFIRM_COUNT = 2
POOL = "stable_history_500d"
PREFIXES = ["600", "601", "603", "605", "000", "001", "002", "003"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_predictions(prediction):
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError(f"预测文件列数不为1: {prediction.shape}")
        prediction = prediction.iloc[:, 0]
    if not isinstance(prediction.index, pd.MultiIndex):
        raise ValueError("预测文件不是日期、股票代码两级索引")
    if set(prediction.index.names) != {"datetime", "instrument"}:
        raise ValueError(f"预测索引层级异常: {prediction.index.names}")
    prediction = prediction.sort_index()
    if prediction.index.has_duplicates:
        raise ValueError("预测文件存在重复的日期、股票键")
    return prediction


def fast_precompute_limit(price_cache, limit_cache, st_codes=None, no_limit_dates=None,
                          st_codes_by_date=None, direction="up"):
    st_codes = st_codes or set()
    no_limit_dates = no_limit_dates or {}
    pit_dates = sorted(st_codes_by_date or {})
    for code, frame in price_cache.items():
        try:
            frame = frame.sort_values("trade_date").reset_index(drop=True)
            price_cache[code] = frame
            trade_dates = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y%m%d").to_numpy()
            closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
            opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float) if "open" in frame.columns else None
            adj = pd.to_numeric(frame["adj_factor"], errors="coerce").to_numpy(dtype=float) if "adj_factor" in frame.columns else None
            code_str = str(code)
            blocked_dates = no_limit_dates.get(code, no_limit_dates.get(code_str, set()))
            limit_dates = {}
            pit_index = -1
            for i in range(1, len(frame)):
                previous_close = closes[i - 1]
                if previous_close <= 0:
                    continue
                date_key = trade_dates[i]
                while pit_index + 1 < len(pit_dates) and pit_dates[pit_index + 1] <= date_key:
                    pit_index += 1
                if date_key in blocked_dates:
                    continue
                is_st = (
                    code_str in st_codes_by_date[pit_dates[pit_index]]
                    if pit_index >= 0 else code_str in st_codes
                )
                if opens is not None:
                    open_price = opens[i]
                    if open_price <= 0 or pd.isna(open_price):
                        continue
                    opening_return = open_price / previous_close
                else:
                    close_price = closes[i]
                    if close_price <= 0:
                        continue
                    opening_return = close_price / previous_close
                if (adj is not None and pd.notna(adj[i]) and pd.notna(adj[i - 1])
                        and adj[i - 1] > 0):
                    opening_return *= adj[i - 1] / adj[i]
                opening_return -= 1
                threshold = backtest_common.limit_pct_for_code(code_str, is_st=is_st) * 0.98
                if (opening_return >= threshold if direction == "up" else opening_return <= -threshold):
                    limit_dates[date_key] = True
            if limit_dates:
                limit_cache[code] = limit_dates
        except Exception:
            pass


def fast_precompute_limit_up(price_cache, limit_cache, st_codes=None, no_limit_dates=None,
                             st_codes_by_date=None):
    fast_precompute_limit(price_cache, limit_cache, st_codes, no_limit_dates,
                          st_codes_by_date, "up")


def fast_precompute_limit_down(price_cache, limit_cache, st_codes=None, no_limit_dates=None,
                               st_codes_by_date=None):
    fast_precompute_limit(price_cache, limit_cache, st_codes, no_limit_dates,
                          st_codes_by_date, "down")


def build_history(prediction):
    dates = sorted(pd.Timestamp(value).normalize() for value in prediction.index.get_level_values("datetime").unique())
    rank_by_date = {}
    score_by_date = {}
    for date in dates:
        day = prediction.xs(date, level="datetime").dropna()
        frame = pd.DataFrame({"score": pd.to_numeric(day, errors="coerce")})
        frame = frame.dropna().reset_index(names="instrument")
        frame["code"] = frame["instrument"].map(norm_code)
        frame = frame.sort_values(["score", "code"], ascending=[False, True], kind="mergesort")
        frame = frame.drop_duplicates("code", keep="first").reset_index(drop=True)
        frame["rank"] = range(1, len(frame) + 1)
        rank_by_date[date.strftime("%Y-%m-%d")] = dict(zip(frame["code"], frame["rank"]))
        score_by_date[date.strftime("%Y-%m-%d")] = dict(zip(frame["code"], frame["score"].astype(float)))

    stable_by_date = {}
    detail_by_date = {}
    date_keys = [date.strftime("%Y-%m-%d") for date in dates]
    for i, date_key in enumerate(date_keys):
        ranks = rank_by_date[date_key]
        count_by_code = {}
        for code in ranks:
            count_by_code[code] = sum(
                rank_by_date[date_keys[j]].get(code, CONFIRM_DEPTH + 1) <= CONFIRM_DEPTH
                for j in range(max(0, i - CONFIRM_WINDOW + 1), i + 1)
            )
        stable = {
            code for code, rank in ranks.items()
            if rank <= CONFIRM_DEPTH and count_by_code.get(code, 0) >= CONFIRM_COUNT
        }
        stable_by_date[date_key] = stable
        detail_by_date[date_key] = count_by_code
    return dates, rank_by_date, score_by_date, stable_by_date, detail_by_date


def decision_dates(metrics):
    return [str(row.get("date"))[:10] for row in metrics.get("periods", [])]


def align_period_returns_to_decisions(equity_curve, periods):
    if len(equity_curve) != len(periods) + 1:
        raise ValueError("净值曲线与调仓期数量不匹配，无法对齐信号与实现收益")
    aligned = {}
    for index, period in enumerate(periods):
        decision_date = str(equity_curve[index].get("date"))[:10]
        period_end_date = str(equity_curve[index + 1].get("date"))[:10]
        if str(period.get("date"))[:10] != period_end_date:
            raise ValueError(f"回测期末日期与净值曲线不一致: {period_end_date}")
        if decision_date in aligned:
            raise ValueError(f"决策日期重复，无法归因实现收益: {decision_date}")
        aligned[decision_date] = {
            "period_return": period.get("period_return"),
            "outcome_end_date": period_end_date,
        }
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default=MODEL_EXP)
    parser.add_argument("--run-id", default=MODEL_RUN)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--decision-end", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    decision_end = args.decision_end or args.end

    mlruns = Path(str(config.QLIB_MLFLOW_URI).replace("file:", ""))
    run_path = mlruns / args.experiment_id / args.run_id
    prediction_path = run_path / "artifacts" / "pred.pkl"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    label_mode = get_model_label_mode(run_path)
    if label_mode != "tradable_open_to_open":
        raise ValueError(f"模型标签合同不匹配: {label_mode}")

    with open(prediction_path, "rb") as source:
        prediction = normalize_predictions(pickle.load(source))
    dates, rank_by_date, score_by_date, stable_by_date, count_by_date = build_history(prediction)
    date_set = {date.strftime("%Y-%m-%d") for date in dates}
    first_date, last_date = min(date_set), max(date_set)
    if first_date > args.start or last_date < decision_end:
        raise ValueError(f"预测日期覆盖不足: {first_date}~{last_date}，请求 {args.start}~{decision_end}")

    output = Path(args.output) if args.output else ROOT / "data" / "experiments" / "rank_persistence_ab_20260925"
    output.mkdir(parents=True, exist_ok=False)
    runner = DirectionBacktestRunner()
    industry_map = DataFetcher().get_industry_map()
    if not industry_map:
        raise RuntimeError("无法取得非空证券名称/行业映射，停止回放以避免静默改变ST过滤")
    DataFetcher.get_industry_map = lambda self: industry_map
    backtest_engine.precompute_limit_up = fast_precompute_limit_up
    backtest_engine.precompute_limit_down = fast_precompute_limit_down
    model_ref = (args.experiment_id, args.run_id)
    cache_key = (
        args.start, args.end, decision_end, POOL, ("trend_momentum",),
        tuple(PREFIXES), (), model_ref,
    )
    runner._cached_run_key = cache_key
    runner._cached_pred_cache = {"fixed": prediction}
    original_filter = runner._filter_decision_day_ranking
    contracts = {
        "start": args.start,
        "end": args.end,
        "decision_end": decision_end,
        "pool": POOL,
        "top_n": TOP_N,
        "frequency": "week",
        "target_exposure": 0.95,
        "max_one_way_turnover": 0.30,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001,
        "stamp_duty": 0.0005,
        "neutralize_qlib": False,
        "decision_day_filter": True,
        "stop_source": "off",
        "macro_bias": False,
        "sector_fusion": False,
        "use_vol_switch": False,
        "code_prefixes": PREFIXES,
        "confirmation": {
            "depth": CONFIRM_DEPTH,
            "prediction_sessions": CONFIRM_WINDOW,
            "required_appearances": CONFIRM_COUNT,
            "definition": "当天及前2个预测交易日中至少2次进入Top20，且当天仍在Top20",
        },
    }
    variants = {}
    audit_rows = []
    filter_reason_adjustments = {}
    variant_modes = ("A_baseline", "B_stability_first", "C_strict_confirm")
    for variant in variant_modes:
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
        unmapped_reasons = {}
        def filtered(ranked_codes, decision_date, target_count, price_cache, sector_map,
                     min_history_observations=None, mode=variant):
            requested_count = target_count if mode == "A_baseline" else max(target_count, 50)
            eligible, excluded, examined = original_filter(
                ranked_codes, decision_date, requested_count, price_cache,
                sector_map, min_history_observations,
            )
            compatible_excluded = {}
            for reason, count in excluded.items():
                if reason in ("st", "suspended", "limit_up", "low_liquidity", "low_price", "new_listing", "data_error"):
                    compatible_excluded[reason] = count
                else:
                    unmapped_reasons[reason] = unmapped_reasons.get(reason, 0) + count
                    compatible_excluded["data_error"] = compatible_excluded.get("data_error", 0) + count
            if mode == "A_baseline":
                return eligible, compatible_excluded, examined
            key = str(decision_date)[:10]
            stable = stable_by_date.get(key, set())
            stable_eligible = [code for code in eligible if norm_code(code) in stable]
            if mode == "C_strict_confirm":
                result_codes = stable_eligible
            else:
                stable_set = set(stable_eligible)
                result_codes = stable_eligible + [code for code in eligible if code not in stable_set]
            return result_codes[:requested_count], compatible_excluded, examined
        runner._filter_decision_day_ranking = filtered

        result = runner.run(
            start=args.start,
            end=args.end,
            decision_end=decision_end,
            pool=POOL,
            top_n=TOP_N,
            freq="week",
            sizing="equal",
            target_exposure=0.95,
            max_one_way_turnover=0.30,
            directions=["trend_momentum"],
            full_position=True,
            code_prefixes=PREFIXES,
            fixed_model=model_ref,
            commission_rate=0.0003,
            slippage_rate=0.0001,
            stamp_duty=0.0005,
            stop_source="off",
            macro_bias=False,
            sector_fusion=False,
            neutralize_qlib=False,
            decision_day_filter=True,
        )
        runner._initial_build = original_initial_build
        if not initial_selection:
            raise RuntimeError(f"{variant}回测没有生成初始组合，拒绝输出逐股归因")
        filter_reason_adjustments[variant] = unmapped_reasons
        excluded_counts = (result.get("score_processing") or {}).get("decision_day_filter_excluded") or {}
        for reason, count in unmapped_reasons.items():
            excluded_counts["data_error"] = max(0, int(excluded_counts.get("data_error", 0)) - count)
            excluded_counts[reason] = count
        result.setdefault("score_processing", {})["decision_day_filter_excluded"] = excluded_counts
        variants[variant] = result
        print(variant, result.get("summary"), flush=True)
        (output / f"partial_{variant}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2,
                       default=lambda value: value.item() if hasattr(value, "item") else str(value)),
            encoding="utf-8",
        )
        returns_by_decision = align_period_returns_to_decisions(
            result.get("equity_curve", []), result.get("periods", [])
        )
        selections = [(initial_selection["date"], initial_selection["codes"])]
        selections.extend(
            (str(period.get("date"))[:10], period.get("top_stocks", [])[:TOP_N])
            for period in result.get("periods", [])
        )
        for date_key, selected_by_rule in selections:
            ranks = rank_by_date.get(date_key, {})
            detail = count_by_date.get(date_key, {})
            outcome = returns_by_decision.get(date_key, {})
            for slot, code in enumerate(selected_by_rule, 1):
                code = norm_code(code)
                audit_rows.append({
                    "variant": variant,
                    "decision_date": date_key,
                    "slot": slot,
                    "code": code,
                    "same_day_rank": ranks.get(code),
                    "appearances_in_window": detail.get(code, 0),
                    "confirmed": code in stable_by_date.get(date_key, set()),
                    "period_return": outcome.get("period_return"),
                    "return_period_end_date": outcome.get("outcome_end_date"),
                })

    runner._filter_decision_day_ranking = original_filter
    report = {
        "status": "exploratory_backtest",
        "interpretation": "历史区间已参与其他模型/组合分析；该次持续性规则结果为探索性，不能用作独立OOS晋级证据。",
        "model": {
            "experiment_id": args.experiment_id,
            "run_id": args.run_id,
            "label_mode": label_mode,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "prediction_rows": int(len(prediction)),
            "prediction_dates": len(dates),
            "first_prediction_date": first_date,
            "last_prediction_date": last_date,
        },
        "contracts": contracts,
        "period_return_alignment": (
            "period return is measured between consecutive next-session-open executions; "
            "the return posted at an end decision date is attributed to the prior decision's selection"
        ),
        "filter_reason_adjustments": filter_reason_adjustments,
        "data_warnings": [
            "历史行业/证券状态快照覆盖不足，回放结果不具备完整点时证券参考数据认证。",
            "本轮不检验基本面公告日效应；fundamental_data公告日覆盖不足。",
            "行情复权因子历史覆盖存在缺口，完整收益回放需复核价格口径。",
        ],
        "variants": variants,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )
    pd.DataFrame(audit_rows).to_csv(output / "selected_stock_audit.csv", index=False, encoding="utf-8-sig")
    print("OUTPUT", output, flush=True)


if __name__ == "__main__":
    main()
