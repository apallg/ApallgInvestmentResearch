import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.adapters.qlib.model_contract import get_model_label_mode
from core.data import Database
from core.data.fetcher import DataFetcher
from core.domain.backtest.common import limit_pct_for_code, norm_code
from core.domain.screening.exclusion import evaluate_decision_day_filter
from core.shared import config


OUTPUT = ROOT / "data" / "experiments" / "top5_top50_hold_20260929"
PROTOCOL = ROOT / "docs" / "top5_top50_hold_protocol_20260929.md"
SEGMENTS = {
    "earlier": {
        "experiment_id": "721018297369182666",
        "run_id": "81161bda8aba4b73ac5fc883e5514439",
        "start": "2024-07-01",
        "end": "2025-06-30",
    },
    "recent": {
        "experiment_id": "656749665990832865",
        "run_id": "7e5e67ba9b714c76b0808a2626d0f7d1",
        "start": "2025-07-01",
        "end": "2026-08-31",
    },
}
VARIANTS = {
    "weekly_top5": {"frequency": "week", "hold_depth": 5},
    "daily_top5": {"frequency": "day", "hold_depth": 5},
    "daily_top50_hold": {"frequency": "day", "hold_depth": 50},
}
TOP_N = 5
TARGET_EXPOSURE = 0.95
RAW_DEPTH = 160
PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
BUY_COST = 0.0004
SELL_COST = 0.0009
DRAW_COUNT = 20000
SEED = 20260929


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
        raise ValueError("预测文件存在重复日期、股票代码")
    return prediction


def build_raw_rankings(prediction, start, end):
    rankings = {}
    scores = {}
    sliced = prediction.loc[pd.Timestamp(start):pd.Timestamp(end)]
    for date, day in sliced.groupby(level="datetime", sort=True):
        series = day.droplevel("datetime").dropna()
        frame = pd.DataFrame({"score": pd.to_numeric(series, errors="coerce")}).dropna()
        frame["code"] = [norm_code(value) for value in frame.index]
        frame = frame[frame["code"].str.startswith(PREFIXES)]
        frame = frame.sort_values(["score", "code"], ascending=[False, True], kind="mergesort")
        frame = frame.drop_duplicates("code", keep="first").head(RAW_DEPTH)
        key = pd.Timestamp(date).strftime("%Y-%m-%d")
        rankings[key] = frame["code"].tolist()
        scores[key] = dict(zip(frame["code"], frame["score"].astype(float)))
    return rankings, scores


def load_calendar(start, end):
    rows = Database.fetchall(
        "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date >= %s AND trade_date <= %s ORDER BY trade_date",
        (start.replace("-", ""), end.replace("-", "")),
    )
    return [pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d") for row in rows]


def load_price_frames(codes, start, end):
    frames = {}
    code_list = sorted(codes)
    for offset in range(0, len(code_list), 250):
        batch = code_list[offset:offset + 250]
        placeholders = ",".join(["%s"] * len(batch))
        rows = Database.fetchall(
            "SELECT code,trade_date,open,close,volume,amount,pct_chg,adj_factor FROM daily_quotes "
            f"WHERE trade_date >= %s AND trade_date <= %s AND code IN ({placeholders}) "
            "ORDER BY code,trade_date",
            [start.replace("-", ""), end.replace("-", "")] + batch,
        )
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        for code, group in frame.groupby("code", sort=False):
            frames[str(code)] = group.sort_values("trade_date").reset_index(drop=True)
        print(f"PRICE_BATCH {min(offset + len(batch), len(code_list))}/{len(code_list)}", flush=True)
    return frames


def latest_adj_factors(codes):
    result = {}
    code_list = sorted(codes)
    for offset in range(0, len(code_list), 500):
        batch = code_list[offset:offset + 500]
        placeholders = ",".join(["%s"] * len(batch))
        rows = Database.fetchall(
            "SELECT q.code,q.adj_factor FROM daily_quotes q INNER JOIN "
            "(SELECT code,MAX(trade_date) trade_date FROM daily_quotes "
            f"WHERE code IN ({placeholders}) GROUP BY code) x "
            "ON q.code=x.code AND q.trade_date=x.trade_date",
            batch,
        )
        for row in rows:
            result[str(row["code"])] = float(row["adj_factor"] or 0)
    return result


def build_eligible_rankings(raw_rankings, price_frames, names, adj_factors):
    eligible = {}
    excluded = {}
    for index, (date, ranked) in enumerate(raw_rankings.items(), 1):
        kept = []
        counts = {}
        for code in ranked:
            info = names.get(code, {})
            name = info.get("name", "") if isinstance(info, dict) else ""
            assessment = evaluate_decision_day_filter(
                code,
                price_frames.get(code),
                date,
                name=name,
                min_turnover=10000000,
                latest_adj_factor=adj_factors.get(code),
                limit_pct=limit_pct_for_code(code, is_st="ST" in str(name).upper()),
                min_history_observations=500,
            )
            reason = assessment["reason"]
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
                continue
            kept.append(code)
            if len(kept) >= 50:
                break
        if len(kept) < 50:
            raise RuntimeError(f"{date} 可交易排名只有 {len(kept)} 只，拒绝日期回退或缩小Top50")
        eligible[date] = kept
        excluded[date] = counts
        if index % 40 == 0:
            print(f"FILTER {index}/{len(raw_rankings)}", flush=True)
    return eligible, excluded


def weekly_decisions(dates):
    available = set(dates)
    start = pd.Timestamp(min(dates))
    end = pd.Timestamp(max(dates))
    result = []
    seen = set()
    for anchor in pd.date_range(start, end, freq="W-THU"):
        for offset in range(7):
            candidate = (anchor - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            if candidate in available and candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
                break
    return result


def row_for_date(frame, date):
    if frame is None or frame.empty:
        return None
    matches = frame[frame["trade_date"] == pd.Timestamp(date)]
    return None if matches.empty else matches.iloc[-1]


def close_on(frame, date):
    if frame is None or frame.empty:
        return None
    rows = frame[frame["trade_date"] <= pd.Timestamp(date)]
    if rows.empty:
        return None
    value = pd.to_numeric(pd.Series([rows.iloc[-1].get("close")]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and value > 0 else None


def close_before(frame, date):
    if frame is None or frame.empty:
        return None
    rows = frame[frame["trade_date"] < pd.Timestamp(date)]
    if rows.empty:
        return None
    value = pd.to_numeric(pd.Series([rows.iloc[-1].get("close")]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and value > 0 else None


def open_limit(frame, date, code, is_st, direction):
    if frame is None or frame.empty:
        return False
    before = frame[frame["trade_date"] <= pd.Timestamp(date)]
    if len(before) < 2 or before.iloc[-1]["trade_date"] != pd.Timestamp(date):
        return False
    current = before.iloc[-1]
    previous = before.iloc[-2]
    open_price = float(current.get("open") or 0)
    previous_close = float(previous.get("close") or 0)
    if open_price <= 0 or previous_close <= 0:
        return False
    raw_ratio = open_price / previous_close
    current_adj = float(current.get("adj_factor") or 0)
    previous_adj = float(previous.get("adj_factor") or 0)
    if current_adj > 0 and previous_adj > 0:
        raw_ratio *= previous_adj / current_adj
    change = raw_ratio - 1
    threshold = limit_pct_for_code(code, is_st=is_st) * 0.98
    return change >= threshold if direction == "up" else change <= -threshold


def simulate(eligible, scores, price_frames, names, calendar, frequency, hold_depth):
    dates = sorted(eligible)
    anchors = set(weekly_decisions(dates))
    next_dates = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    cash = 1.0
    holdings = {}
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
        rebalance = frequency == "day" or decision_date in anchors
        open_nav = cash
        for code, position in holdings.items():
            row = row_for_date(price_frames.get(code), execution_date)
            price = float(row.get("open") or 0) if row is not None else 0
            if price <= 0:
                price = close_before(price_frames.get(code), execution_date) or position["last_price"]
            open_nav += position["shares"] * price
        if rebalance:
            hold_set = set(eligible[decision_date][:hold_depth])
            for code in list(holdings):
                if code in hold_set:
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
                    "gross": gross,
                    "cost": fee,
                    "trade_return": trade_return,
                })
                del holdings[code]
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
    peak = np.maximum.accumulate(values)
    drawdown = values / peak - 1
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


def anchor_returns(result, anchors, next_dates):
    equity = {row["date"]: float(row["equity"]) for row in result["equity"]}
    rows = []
    previous = None
    for decision_date in anchors:
        execution_date = next_dates[decision_date]
        value = equity.get(execution_date)
        if value is None:
            raise RuntimeError(f"{decision_date}/{execution_date} 缺少锚点净值")
        if previous is not None:
            rows.append({
                "decision_date": decision_date,
                "outcome_date": execution_date,
                "period_return": value / previous - 1,
            })
        previous = value
    return rows


def compare(segments, baseline, idea):
    arrays = []
    for segment in segments.values():
        a = np.array([row["period_return"] for row in segment["weekly_returns"][baseline]])
        b = np.array([row["period_return"] for row in segment["weekly_returns"][idea]])
        if len(a) != len(b):
            raise RuntimeError("配对周长度不一致")
        arrays.append((a, b))
    a_all = np.concatenate([row[0] for row in arrays])
    b_all = np.concatenate([row[1] for row in arrays])
    difference = b_all - a_all
    wins = int((difference > 0).sum())
    losses = int((difference < 0).sum())
    non_ties = wins + losses
    result = {
        "weeks": len(a_all),
        "baseline_profitable_weeks": int((a_all > 0).sum()),
        "idea_profitable_weeks": int((b_all > 0).sum()),
        "baseline_profit_win_rate": float((a_all > 0).mean()),
        "idea_profit_win_rate": float((b_all > 0).mean()),
        "profit_win_rate_difference_pp": float(((b_all > 0).mean() - (a_all > 0).mean()) * 100),
        "idea_beats_baseline_weeks": wins,
        "idea_loses_to_baseline_weeks": losses,
        "ties": int(len(difference) - non_ties),
        "idea_outperformance_share_non_ties": wins / non_ties if non_ties else None,
        "mean_weekly_return_difference_pp": float(difference.mean() * 100),
        "bootstrap": {},
    }
    for block_size in (4, 8, 13):
        rng = np.random.default_rng(SEED)
        sampled_a = []
        sampled_b = []
        for a, b in arrays:
            starts = rng.integers(0, len(a), size=(DRAW_COUNT, math.ceil(len(a) / block_size)))
            indices = ((starts[:, :, None] + np.arange(block_size)) % len(a)).reshape(DRAW_COUNT, -1)[:, :len(a)]
            sampled_a.append(a[indices])
            sampled_b.append(b[indices])
        sample_a = np.concatenate(sampled_a, axis=1)
        sample_b = np.concatenate(sampled_b, axis=1)
        sample_diff = sample_b - sample_a
        sample_profit_diff = (sample_b > 0).mean(axis=1) - (sample_a > 0).mean(axis=1)
        sample_wins = (sample_diff > 0).sum(axis=1)
        sample_non_ties = (sample_diff != 0).sum(axis=1)
        active = sample_non_ties > 0
        result["bootstrap"][str(block_size)] = {
            "profit_win_rate_difference_ci95_pp": (np.quantile(sample_profit_diff, [0.025, 0.975]) * 100).tolist(),
            "outperformance_share_ci95": np.quantile(
                sample_wins[active] / sample_non_ties[active], [0.025, 0.975]
            ).tolist(),
            "mean_weekly_return_difference_ci95_pp": (
                np.quantile(sample_diff.mean(axis=1), [0.025, 0.975]) * 100
            ).tolist(),
        }
    result["profit_win_rate_evidence_pass"] = all(
        row["profit_win_rate_difference_ci95_pp"][0] > 0 for row in result["bootstrap"].values()
    )
    result["outperformance_evidence_pass"] = all(
        row["outperformance_share_ci95"][0] > 0.5 for row in result["bootstrap"].values()
    )
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {OUTPUT}")
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
            missing = sorted(set(expected_dates) - set(raw_rankings))
            extra = sorted(set(raw_rankings) - set(expected_dates))
            raise RuntimeError(f"{segment_name} 预测交易日不完整: missing={missing[:3]}, extra={extra[:3]}")
        all_codes = {code for ranked in raw_rankings.values() for code in ranked}
        load_start = (pd.Timestamp(contract["start"]) - pd.Timedelta(days=1200)).strftime("%Y-%m-%d")
        calendar_end = (pd.Timestamp(contract["end"]) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        calendar = load_calendar(load_start, calendar_end)
        next_dates = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
        execution_end = next_dates.get(contract["end"])
        if not execution_end:
            raise RuntimeError(f"{segment_name} 末日缺少下一交易日")
        price_frames = load_price_frames(all_codes, load_start, execution_end)
        adj_factors = latest_adj_factors(all_codes)
        eligible, excluded = build_eligible_rankings(raw_rankings, price_frames, names, adj_factors)
        anchors = weekly_decisions(sorted(eligible))
        variants = {}
        weekly_returns = {}
        for variant, settings in VARIANTS.items():
            print(f"SIMULATE {segment_name} {variant}", flush=True)
            variants[variant] = simulate(
                eligible,
                scores,
                price_frames,
                names,
                calendar,
                settings["frequency"],
                settings["hold_depth"],
            )
            weekly_returns[variant] = anchor_returns(variants[variant], anchors, next_dates)
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
                "factor_keys_sha256": sha256(prediction_path.parent / "factor_keys.json"),
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
        "idea_vs_daily_top5": compare(segment_results, "daily_top5", "daily_top50_hold"),
        "idea_vs_weekly_top5": compare(segment_results, "weekly_top5", "daily_top50_hold"),
    }
    report = {
        "status": "exploratory_historical_replay",
        "interpretation": "已查看历史上的探索性复验；不构成独立OOS或生产晋级证据。",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
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
            "历史证券名称、ST、行业和完整上市状态没有逐日PIT快照。",
            "档案模型及样本日期已在先前研究中查看，本结果不是独立确认性OOS。",
            "回放未建模开盘集合竞价容量、最低佣金和冲击成本。",
        ],
        "code_sha256": sha256(Path(__file__)),
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
