import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.adapters.qlib.model_contract import get_model_label_mode
from core.data.fetcher import DataFetcher
from core.shared import config
import scripts.analyze_top5_top50_hold as base_experiment
from scripts.analyze_top5_top50_hold import (
    BUY_COST,
    PREFIXES,
    RAW_DEPTH,
    SEGMENTS,
    SELL_COST,
    TARGET_EXPOSURE,
    TOP_N,
    anchor_returns,
    build_eligible_rankings,
    build_raw_rankings,
    close_before,
    close_on,
    compare,
    latest_adj_factors,
    load_calendar,
    load_price_frames,
    normalize_predictions,
    open_limit,
    row_for_date,
    sha256,
    simulate,
    weekly_decisions,
)


OUTPUT = ROOT / "data" / "experiments" / "top50_exit_grace_20260930"
PROTOCOL = ROOT / "docs" / "top50_exit_grace_protocol_20260930.md"
SOURCE_REPORT = ROOT / "data" / "experiments" / "top5_top50_hold_20260929" / "report.json"
VARIANTS = {
    "weekly_top5": {"frequency": "week", "grace_days": None},
    "top50_exit_1d": {"frequency": "day", "grace_days": 1},
    "top50_exit_5d": {"frequency": "day", "grace_days": 5},
    "top50_exit_10d": {"frequency": "day", "grace_days": 10},
}
base_experiment.SEED = 20260930


def simulate_grace(eligible, price_frames, names, calendar, grace_days):
    dates = sorted(eligible)
    next_dates = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    cash = 1.0
    holdings = {}
    outside_streak = {}
    equity_rows = []
    trades = []
    turnover_notional = 0.0
    costs = 0.0
    blocked_buys = 0
    blocked_sells = 0
    for decision_date in dates:
        execution_date = next_dates.get(decision_date)
        if not execution_date:
            raise RuntimeError(f"{decision_date} 缺少下一交易日")
        open_nav = cash
        for code, position in holdings.items():
            row = row_for_date(price_frames.get(code), execution_date)
            price = float(row.get("open") or 0) if row is not None else 0
            if price <= 0:
                price = close_before(price_frames.get(code), execution_date) or position["last_price"]
            open_nav += position["shares"] * price
        hold_set = set(eligible[decision_date][:50])
        for code in list(holdings):
            outside_streak[code] = 0 if code in hold_set else outside_streak.get(code, 0) + 1
            if outside_streak[code] < grace_days:
                continue
            frame = price_frames.get(code)
            row = row_for_date(frame, execution_date)
            info = names.get(code, {})
            name = info.get("name", "") if isinstance(info, dict) else ""
            if row is None or float(row.get("open") or 0) <= 0 or open_limit(
                    frame, execution_date, code, "ST" in str(name).upper(), "down"):
                blocked_sells += 1
                continue
            price = float(row["open"])
            gross = holdings[code]["shares"] * price
            fee = gross * SELL_COST
            cash += gross - fee
            turnover_notional += gross
            costs += fee
            trade_return = (gross - fee) / holdings[code]["buy_total"] - 1
            trades.append({
                "decision_date": decision_date,
                "execution_date": execution_date,
                "code": code,
                "action": "sell",
                "rank_exit": next((i + 1 for i, value in enumerate(eligible[decision_date]) if value == code), None),
                "outside_top50_days": outside_streak[code],
                "gross": gross,
                "cost": fee,
                "trade_return": trade_return,
            })
            del holdings[code]
            del outside_streak[code]
        vacancies = TOP_N - len(holdings)
        slot_notional = open_nav * TARGET_EXPOSURE / TOP_N
        for code in eligible[decision_date][:TOP_N]:
            if vacancies <= 0:
                break
            if code in holdings:
                continue
            frame = price_frames.get(code)
            row = row_for_date(frame, execution_date)
            info = names.get(code, {})
            name = info.get("name", "") if isinstance(info, dict) else ""
            if row is None or float(row.get("open") or 0) <= 0 or open_limit(
                    frame, execution_date, code, "ST" in str(name).upper(), "up"):
                blocked_buys += 1
                continue
            price = float(row["open"])
            gross = min(slot_notional, cash / (1 + BUY_COST))
            if gross <= 0:
                continue
            fee = gross * BUY_COST
            cash -= gross + fee
            holdings[code] = {
                "shares": gross / price,
                "buy_total": gross + fee,
                "last_price": price,
                "entry_date": execution_date,
            }
            outside_streak[code] = 0
            turnover_notional += gross
            costs += fee
            trades.append({
                "decision_date": decision_date,
                "execution_date": execution_date,
                "code": code,
                "action": "buy",
                "rank_entry": eligible[decision_date].index(code) + 1,
                "gross": gross,
                "cost": fee,
            })
            vacancies -= 1
        close_nav = cash
        for code, position in holdings.items():
            price = close_on(price_frames.get(code), execution_date) or position["last_price"]
            position["last_price"] = price
            close_nav += position["shares"] * price
        equity_rows.append({
            "decision_date": decision_date,
            "date": execution_date,
            "equity": close_nav,
            "cash": cash,
            "holdings": len(holdings),
        })
    equity = pd.DataFrame(equity_rows)
    values = equity["equity"].to_numpy(dtype=float)
    daily_returns = pd.Series(values).pct_change(fill_method=None).dropna().to_numpy()
    drawdown = values / np.maximum.accumulate(values) - 1
    completed = [row["trade_return"] for row in trades if row["action"] == "sell"]
    return {
        "equity": equity_rows,
        "trades": trades,
        "summary": {
            "total_return": float(values[-1] - 1),
            "max_drawdown": float(drawdown.min()),
            "daily_profitable_rate": float((daily_returns > 0).mean()),
            "daily_observations": int(len(daily_returns)),
            "completed_trades": len(completed),
            "completed_trade_win_rate": float((np.array(completed) > 0).mean()) if completed else None,
            "turnover_notional_over_initial_nav": float(turnover_notional),
            "cost_over_initial_nav": float(costs),
            "blocked_buys": blocked_buys,
            "blocked_sells": blocked_sells,
            "average_holdings": float(equity["holdings"].mean()),
            "average_cash_ratio": float((equity["cash"] / equity["equity"]).mean()),
        },
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {OUTPUT}")
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    mlruns = Path(str(config.QLIB_MLFLOW_URI).replace("file:", ""))
    names = DataFetcher().get_industry_map()
    if not names:
        raise RuntimeError("无法取得非空证券名称映射")
    segment_results = {}
    weekly_rows = []
    for segment_name, contract in SEGMENTS.items():
        print(f"SEGMENT {segment_name}", flush=True)
        run_path = mlruns / contract["experiment_id"] / contract["run_id"]
        prediction_path = run_path / "artifacts" / "pred.pkl"
        if get_model_label_mode(run_path) != "tradable_open_to_open":
            raise RuntimeError(f"{segment_name} 模型标签合同不匹配")
        with open(prediction_path, "rb") as source:
            prediction = normalize_predictions(pickle.load(source))
        raw_rankings, scores = build_raw_rankings(prediction, contract["start"], contract["end"])
        expected_dates = load_calendar(contract["start"], contract["end"])
        if sorted(raw_rankings) != expected_dates:
            raise RuntimeError(f"{segment_name} 预测交易日不完整")
        all_codes = {code for ranked in raw_rankings.values() for code in ranked}
        load_start = (pd.Timestamp(contract["start"]) - pd.Timedelta(days=1200)).strftime("%Y-%m-%d")
        calendar_end = (pd.Timestamp(contract["end"]) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        calendar = load_calendar(load_start, calendar_end)
        next_dates = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
        execution_end = next_dates.get(contract["end"])
        if not execution_end:
            raise RuntimeError(f"{segment_name} 末日缺少下一交易日")
        price_frames = load_price_frames(all_codes, load_start, execution_end)
        eligible, excluded = build_eligible_rankings(
            raw_rankings, price_frames, names, latest_adj_factors(all_codes)
        )
        anchors = weekly_decisions(sorted(eligible))
        variants = {}
        weekly_returns = {}
        for variant, settings in VARIANTS.items():
            print(f"SIMULATE {segment_name} {variant}", flush=True)
            if variant == "weekly_top5":
                result = simulate(eligible, scores, price_frames, names, calendar, "week", 5)
            else:
                result = simulate_grace(
                    eligible, price_frames, names, calendar, settings["grace_days"]
                )
            variants[variant] = result
            weekly_returns[variant] = anchor_returns(result, anchors, next_dates)
        source_variant = source_report["segments"][segment_name]["variants"]["daily_top50_hold"]
        current_equity = np.array([row["equity"] for row in variants["top50_exit_1d"]["equity"]])
        source_equity = np.array([row["equity"] for row in source_variant["equity"]])
        if not np.array_equal(current_equity, source_equity):
            raise RuntimeError(f"{segment_name} 1日规则未精确复现前一实验")
        for index in range(len(weekly_returns["weekly_top5"])):
            row = {
                "segment": segment_name,
                "decision_date": weekly_returns["weekly_top5"][index]["decision_date"],
                "outcome_date": weekly_returns["weekly_top5"][index]["outcome_date"],
            }
            for variant in VARIANTS:
                row[variant] = weekly_returns[variant][index]["period_return"]
            weekly_rows.append(row)
        segment_results[segment_name] = {
            "contract": contract,
            "model": {
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "prediction_rows": len(prediction),
                "prediction_dates": len(raw_rankings),
            },
            "execution_end": execution_end,
            "eligible_dates": len(eligible),
            "weekly_anchors": len(anchors),
            "filter_excluded_totals": {
                reason: sum(row.get(reason, 0) for row in excluded.values())
                for reason in sorted({key for row in excluded.values() for key in row})
            },
            "variants": variants,
            "weekly_returns": weekly_returns,
        }
    comparisons = {
        "exit_5d_vs_1d": compare(segment_results, "top50_exit_1d", "top50_exit_5d"),
        "exit_10d_vs_1d": compare(segment_results, "top50_exit_1d", "top50_exit_10d"),
        "exit_10d_vs_5d": compare(segment_results, "top50_exit_5d", "top50_exit_10d"),
        "exit_5d_vs_weekly_top5": compare(segment_results, "weekly_top5", "top50_exit_5d"),
        "exit_10d_vs_weekly_top5": compare(segment_results, "weekly_top5", "top50_exit_10d"),
    }
    report = {
        "status": "secondary_exploratory_historical_replay",
        "interpretation": "在查看1日Top50结果后提出的二次探索；不构成独立OOS或生产晋级证据。",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "source_report": str(SOURCE_REPORT),
        "source_report_sha256": sha256(SOURCE_REPORT),
        "contracts": {
            "top_n": TOP_N,
            "target_exposure": TARGET_EXPOSURE,
            "buy_cost": BUY_COST,
            "sell_cost": SELL_COST,
            "raw_candidate_depth": RAW_DEPTH,
            "code_prefixes": PREFIXES,
            "decision_time": "t_close",
            "execution_time": "t_plus_1_open",
            "valuation_time": "t_plus_1_close",
            "variants": VARIANTS,
        },
        "comparisons": comparisons,
        "segments": segment_results,
        "data_warnings": [
            "规则在查看前一轮结果后提出，存在二次探索和数据窥探风险。",
            "历史证券名称、ST、行业和完整上市状态没有逐日PIT快照。",
            "回放未建模开盘集合竞价容量、最低佣金和额外冲击成本。",
        ],
        "code_sha256": {
            "scripts/analyze_top50_exit_grace.py": sha256(Path(__file__)),
            "scripts/analyze_top5_top50_hold.py": sha256(ROOT / "scripts" / "analyze_top5_top50_hold.py"),
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(weekly_rows).to_csv(OUTPUT / "weekly_pairs.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "comparisons": comparisons,
        "summaries": {
            segment: {variant: data["summary"] for variant, data in result["variants"].items()}
            for segment, result in segment_results.items()
        },
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
