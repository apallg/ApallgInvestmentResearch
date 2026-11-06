import datetime
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data import Database

parser = argparse.ArgumentParser()
parser.add_argument("--start-date", default="2022-07-01")
parser.add_argument("--end-date", default="2026-09-15")
parser.add_argument("--output-dir", default="data/experiments/stock_context_p1_w8_qfqfix_20261020")
args = parser.parse_args()
START = pd.Timestamp(args.start_date)
END = pd.Timestamp(args.end_date)
if START > END:
    raise ValueError("start-date 不能大于 end-date")
WARMUP_DAYS = 180
OUTPUT_DIR = Path(args.output_dir)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = ROOT / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

groups = {
    "金融地产": {"银行", "非银金融", "房地产"},
    "资源材料": {"煤炭", "石油石化", "有色金属", "钢铁", "基础化工", "建筑材料"},
    "基建公用": {"公用事业", "交通运输", "环保", "建筑装饰"},
    "工业制造": {"机械设备", "电力设备", "国防军工", "汽车", "轻工制造"},
    "消费医疗": {"食品饮料", "农林牧渔", "家用电器", "纺织服饰", "商贸零售", "社会服务", "美容护理", "医药生物"},
    "科技传媒": {"电子", "计算机", "通信", "传媒"},
}
group_names = list(groups)
group_ids = {name: i for i, name in enumerate(group_names)}
context_columns = (
    [f"context_group_{i}" for i in range(7)]
    + ["context_log_total_mv", "context_log_circ_mv", "context_turnover", "context_log_amount",
       "context_log_listed_age", "context_board_main", "context_board_star", "context_board_chinext", "context_board_beijing",
       "context_market_ret20", "context_market_ret60", "context_market_vol20", "context_market_vol60", "context_market_breadth",
       "context_beta60", "context_beta120", "context_corr60", "context_corr120", "context_residvol60", "context_residvol120",
       "context_turnover_mean20", "context_turnover_mean60", "context_log_amount_mean20", "context_log_amount_mean60",
       "context_industry_rel20", "context_industry_rel60", "context_industry_vol20", "context_industry_vol60",
       "context_missing_size", "context_missing_liquidity", "context_missing_market_relation", "context_missing_industry_relation"]
)

universe = {}
for line in (ROOT / "data/qlib_data/instruments/all.txt").read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) >= 3:
        universe[parts[0].lower()] = (pd.Timestamp(parts[1]), pd.Timestamp(parts[2]))
symbols = {code[2:] for code in universe if code.startswith(("sh", "sz", "bj"))}
identity = Database.fetchall("SELECT ts_code,symbol,list_date FROM stock_identity")
identity_by_symbol = {row["symbol"]: row for row in identity}
instrument_by_symbol = {}
for symbol, row in identity_by_symbol.items():
    _, suffix = row["ts_code"].split(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix)
    if prefix:
        instrument_by_symbol[symbol] = prefix + symbol
membership = Database.fetchall("SELECT ts_code,index_code,index_name,in_date,out_date FROM stock_index_membership ORDER BY ts_code,in_date")
membership_by_symbol = {}
for row in membership:
    symbol = row["ts_code"].split(".")[0]
    group_id = next((i for i, names in enumerate(groups.values()) if row["index_name"] in names), 6)
    membership_by_symbol.setdefault(symbol, []).append((row["in_date"], row["out_date"], row["index_code"], group_id))
membership_interval_conflicts = 0
for intervals in membership_by_symbol.values():
    for index, (start, end, industry_code, _) in enumerate(intervals):
        for earlier_start, earlier_end, earlier_code, _ in intervals[:index]:
            if industry_code != earlier_code and start <= (earlier_end or datetime.date.max) and earlier_start <= (end or datetime.date.max):
                membership_interval_conflicts += 1
if membership_interval_conflicts:
    raise RuntimeError(f"历史行业区间冲突: {membership_interval_conflicts}")

calendar = (ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()
benchmark_values = np.fromfile(ROOT / "data/qlib_data/features/sh000300/close.day.bin", dtype="<f4")
benchmark_start = int(benchmark_values[0])
benchmark_prices = benchmark_values[1:]
benchmark_dates = pd.to_datetime(calendar[benchmark_start:benchmark_start + len(benchmark_prices)])
benchmark = pd.Series(benchmark_prices, index=benchmark_dates).replace([np.inf, -np.inf], np.nan).dropna()
benchmark_return = benchmark.div(benchmark.shift(1)).sub(1.0)
benchmark_return.name = "benchmark_return"
trading_day_positions = pd.Series(np.arange(len(calendar)), index=pd.to_datetime(calendar))

chunks = []
coverage = []
cursor = START
while cursor <= END:
    chunk_end = min(cursor + pd.Timedelta(days=179), END)
    query_start = max(pd.Timestamp("2012-01-01"), cursor - pd.Timedelta(days=WARMUP_DAYS))
    placeholders = ",".join(["%s"] * len(symbols))
    query = (
        "SELECT q.code,q.trade_date,q.close,q.adj_factor,q.amount,d.total_mv,d.circ_mv,d.turnover_rate "
        "FROM daily_quotes q LEFT JOIN daily_basic d ON d.code=q.code AND d.trade_date=q.trade_date "
        "WHERE q.trade_date BETWEEN %s AND %s AND q.code IN (" + placeholders + ") ORDER BY q.code,q.trade_date"
    )
    rows = Database.fetchall(query, (query_start.date(), chunk_end.date(), *sorted(symbols)))
    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError(f"上下文原始行情为空: {query_start.date()}..{chunk_end.date()}")
    print(json.dumps({"stage": "loaded", "start": str(query_start.date()), "end": str(chunk_end.date()), "rows": len(data)}, ensure_ascii=False), flush=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for column in ("close", "adj_factor", "amount", "total_mv", "circ_mv", "turnover_rate"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["adj_close"] = data["close"]
    previous_adj_close = data.groupby("code", sort=False)["adj_close"].shift(1)
    data["trading_day_position"] = data["trade_date"].map(trading_day_positions)
    previous_trading_day_position = data.groupby("code", sort=False)["trading_day_position"].shift(1)
    data["stock_return"] = data["adj_close"].div(previous_adj_close).sub(1.0)
    data.loc[(data["trading_day_position"] - previous_trading_day_position != 1) | ~np.isfinite(data["stock_return"]), "stock_return"] = np.nan

    market_daily = data.groupby("trade_date", sort=True)["stock_return"].agg(
        market_return="mean", market_breadth=lambda values: float((values.dropna() > 0).mean()) if values.notna().any() else np.nan
    )
    market_daily = market_daily.join(benchmark_return, how="left")
    market_daily["market_ret20"] = (1.0 + market_daily["market_return"]).rolling(20, min_periods=18).apply(np.prod, raw=True) - 1.0
    market_daily["market_ret60"] = (1.0 + market_daily["market_return"]).rolling(60, min_periods=54).apply(np.prod, raw=True) - 1.0
    market_daily["market_vol20"] = market_daily["market_return"].rolling(20, min_periods=18).std()
    market_daily["market_vol60"] = market_daily["market_return"].rolling(60, min_periods=54).std()
    market_daily["benchmark_return"] = pd.to_numeric(market_daily["benchmark_return"], errors="coerce")

    code_ids = {symbol: {"ts_code": row["ts_code"], "list_date": row["list_date"], "instrument": instrument_by_symbol.get(symbol)} for symbol, row in identity_by_symbol.items()}
    data["industry_code"] = ""
    data["group_id"] = np.uint8(6)
    positions_by_code = data.groupby("code", sort=False).indices
    date_array = data["trade_date"].to_numpy(dtype="datetime64[D]")
    industry_values = data["industry_code"].to_numpy(dtype=object)
    group_values = data["group_id"].to_numpy(dtype=np.uint8)
    for symbol, positions in positions_by_code.items():
        intervals = membership_by_symbol.get(str(symbol), ())
        if not intervals:
            continue
        stock_dates = date_array[positions]
        for in_date, out_date, industry_code, group_id in intervals:
            start_pos = np.searchsorted(stock_dates, np.datetime64(in_date, "D"), side="right")
            end_pos = len(stock_dates) if out_date is None else np.searchsorted(stock_dates, np.datetime64(out_date, "D"), side="right")
            if end_pos > start_pos:
                selected = positions[start_pos:end_pos]
                industry_values[selected] = industry_code
                group_values[selected] = group_id
    data["industry_code"] = industry_values
    data["group_id"] = group_values
    industry_daily = data.loc[data["industry_code"] != ""].groupby(["trade_date", "industry_code"], sort=False)["stock_return"].agg(["sum", "count"])
    row_keys = pd.MultiIndex.from_arrays([data["trade_date"], data["industry_code"]])
    matched_industry = industry_daily.reindex(row_keys)
    industry_sum = matched_industry["sum"].to_numpy(dtype=float)
    industry_count = matched_industry["count"].to_numpy(dtype=float)
    stock_return = data["stock_return"].to_numpy(dtype=float)
    data["industry_loo_return"] = np.where(industry_count > 1, (industry_sum - stock_return) / (industry_count - 1), np.nan)
    data["instrument"] = data["code"].map(lambda code: code_ids.get(str(code), {}).get("instrument"))
    data["list_date"] = data["code"].map(lambda code: code_ids.get(str(code), {}).get("list_date"))
    data["log_total_mv"] = np.log1p(data["total_mv"].where(data["total_mv"] >= 0))
    data["log_circ_mv"] = np.log1p(data["circ_mv"].where(data["circ_mv"] >= 0))
    data["log_amount"] = np.log1p(data["amount"].where(data["amount"] >= 0))
    data["turnover"] = data["turnover_rate"] / 100.0
    data["log_listed_age"] = np.log1p((data["trade_date"] - pd.to_datetime(data["list_date"], errors="coerce")).dt.days.clip(lower=0))
    data["board"] = np.select(
        [data["instrument"].astype(str).str.startswith("bj"), data["instrument"].astype(str).str.startswith("sh68"), data["instrument"].astype(str).str.startswith("sz30")],
        [3, 1, 2], default=0,
    ).astype(np.uint8)

    market_features = market_daily[["market_ret20", "market_ret60", "market_vol20", "market_vol60", "market_breadth"]]
    data = data.join(market_features, on="trade_date")
    data = data.join(market_daily["benchmark_return"], on="trade_date")
    data["context_market_return"] = data["benchmark_return"]
    del data["benchmark_return"]

    grouped = data.groupby("code", sort=False)
    rolling = lambda column, window, min_periods, operation: getattr(grouped[column].rolling(window, min_periods=min_periods), operation)().reset_index(level=0, drop=True).reindex(data.index)
    stock_ret = data["stock_return"]
    market_ret = data["context_market_return"]
    paired = stock_ret.notna() & market_ret.notna()
    data["paired_stock_return"] = stock_ret.where(paired)
    data["paired_market_return"] = market_ret.where(paired)
    data["paired_product"] = (stock_ret * market_ret).where(paired)
    data["paired_stock_square"] = stock_ret.pow(2).where(paired)
    data["paired_market_square"] = market_ret.pow(2).where(paired)
    for window, min_periods in ((60, 54), (120, 108)):
        mean_stock = rolling("paired_stock_return", window, min_periods, "mean")
        mean_market = rolling("paired_market_return", window, min_periods, "mean")
        mean_product = rolling("paired_product", window, min_periods, "mean")
        mean_stock_sq = rolling("paired_stock_square", window, min_periods, "mean")
        mean_market_sq = rolling("paired_market_square", window, min_periods, "mean")
        covariance = mean_product - mean_stock * mean_market
        variance_stock = (mean_stock_sq - mean_stock.pow(2)).clip(lower=0)
        variance_market = (mean_market_sq - mean_market.pow(2)).clip(lower=0)
        data[f"beta{window}"] = covariance.div(variance_market.where(variance_market > 1e-12))
        data[f"corr{window}"] = covariance.div(np.sqrt(variance_stock * variance_market).where((variance_stock * variance_market) > 1e-20))
        residual_variance = (variance_stock - covariance.pow(2).div(variance_market.where(variance_market > 1e-12))).clip(lower=0)
        data[f"residvol{window}"] = np.sqrt(residual_variance)
    for window, min_periods in ((20, 18), (60, 54)):
        data[f"turnover_mean{window}"] = rolling("turnover", window, min_periods, "mean")
        data[f"log_amount_mean{window}"] = rolling("log_amount", window, min_periods, "mean")
        industry_logret = np.log1p(data["industry_loo_return"].clip(lower=-0.999999))
        stock_logret = np.log1p(stock_ret.clip(lower=-0.999999))
        industry_sum = industry_logret.groupby(data["code"], sort=False).rolling(window, min_periods=min_periods).sum().reset_index(level=0, drop=True).reindex(data.index)
        stock_sum = stock_logret.groupby(data["code"], sort=False).rolling(window, min_periods=min_periods).sum().reset_index(level=0, drop=True).reindex(data.index)
        data[f"industry_rel{window}"] = np.expm1(industry_sum - stock_sum)
        data[f"industry_vol{window}"] = rolling("industry_loo_return", window, min_periods, "std")

    for i in range(7):
        data[f"context_group_{i}"] = (data["group_id"] == i).astype(np.float32)
    data["context_log_total_mv"] = data["log_total_mv"]
    data["context_log_circ_mv"] = data["log_circ_mv"]
    data["context_turnover"] = data["turnover"]
    data["context_log_amount"] = data["log_amount"]
    data["context_log_listed_age"] = data["log_listed_age"]
    for i, name in enumerate(("main", "star", "chinext", "beijing")):
        data[f"context_board_{name}"] = (data["board"] == i).astype(np.float32)
    data["context_market_ret20"] = data["market_ret20"]
    data["context_market_ret60"] = data["market_ret60"]
    data["context_market_vol20"] = data["market_vol20"]
    data["context_market_vol60"] = data["market_vol60"]
    data["context_market_breadth"] = data["market_breadth"]
    for name in ("beta60", "beta120", "corr60", "corr120", "residvol60", "residvol120", "turnover_mean20", "turnover_mean60", "log_amount_mean20", "log_amount_mean60", "industry_rel20", "industry_rel60", "industry_vol20", "industry_vol60"):
        data[f"context_{name}"] = data[name]
    data["context_missing_size"] = data[["log_total_mv", "log_circ_mv"]].isna().any(axis=1).astype(np.float32)
    data["context_missing_liquidity"] = data[["turnover", "log_amount"]].isna().any(axis=1).astype(np.float32)
    data["context_missing_market_relation"] = data[["beta60", "beta120", "corr60", "corr120", "residvol60", "residvol120"]].isna().any(axis=1).astype(np.float32)
    data["context_missing_industry_relation"] = data[["industry_rel20", "industry_rel60", "industry_vol20", "industry_vol60"]].isna().any(axis=1).astype(np.float32)
    data = data.loc[(data["trade_date"] >= cursor) & (data["trade_date"] <= chunk_end) & data["instrument"].notna()].copy()
    data["datetime"] = data["trade_date"]
    absent = set(context_columns) - set(data.columns)
    if absent:
        raise RuntimeError(f"P1特征未生成: {sorted(absent)}")
    for column in context_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = data[column].isna()
        if column not in ("context_missing_size", "context_missing_liquidity", "context_missing_market_relation", "context_missing_industry_relation"):
            data.loc[missing, column] = 0.0
    output_columns = ["datetime", "instrument", "group_id", *context_columns]
    result = data[output_columns].set_index(["datetime", "instrument"]).sort_index()
    coverage.append({"start": str(cursor.date()), "end": str(chunk_end.date()), "rows": int(len(result)), "unknown_share": float((result["group_id"] == 6).mean()) if len(result) else None})
    chunks.append(result)
    print(json.dumps({"stage": "chunk_complete", "start": str(cursor.date()), "end": str(chunk_end.date()), "rows": len(result)}, ensure_ascii=False), flush=True)
    del data, result, rows
    cursor = chunk_end + pd.Timedelta(days=1)

panel = pd.concat(chunks).sort_index()
if panel.index.has_duplicates:
    raise RuntimeError("P1上下文面板存在重复日期股票")
missing_rates = panel[["context_missing_size", "context_missing_liquidity", "context_missing_market_relation", "context_missing_industry_relation"]].mean().to_dict()
for column in ("context_market_ret20", "context_market_ret60"):
    if panel[column].nunique(dropna=True) <= 1:
        raise RuntimeError(f"P1市场特征异常恒定: {column}")
panel.to_parquet(OUTPUT_DIR / "context_features.parquet", compression="zstd")
report = {
    "status": "built",
    "feature_contract_version": 2,
    "history_denominator_verified": False,
    "stockyear_coverage_verified": False,
    "industry_pit_verified": False,
    "as_of": datetime.date.today().isoformat(),
    "feature_columns": list(context_columns),
    "rows": int(len(panel)),
    "instruments": int(panel.index.get_level_values("instrument").nunique()),
    "first_date": str(panel.index.get_level_values("datetime").min().date()),
    "last_date": str(panel.index.get_level_values("datetime").max().date()),
    "missing_indicator_rate_before_value_fill": missing_rates,
    "missing_indicator_columns": ["context_missing_size", "context_missing_liquidity", "context_missing_market_relation", "context_missing_industry_relation"],
    "unknown_group_share": float((panel["group_id"] == 6).mean()),
    "membership_interval_conflicts": membership_interval_conflicts,
    "industry_vintage": "effective-date reconstruction; source publication vintages unavailable",
    "return_source": "daily_quotes.close is already qfq; use close directly without multiplying adj_factor; market proxy is Qlib sh000300 index close",
    "chunk_coverage": coverage,
}
(OUTPUT_DIR / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps({"output": str(OUTPUT_DIR), "report": report}, ensure_ascii=False, default=str))
