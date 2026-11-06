"""冻结因子上的专家训练方式与市场路由严格 walk-forward 实验。"""
import argparse
import gc
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service.instrument_resolver import build_filtered_instruments
from backend.service.expert_training import compute_state_affinity
from core.adapters.qlib.config import init_qlib
from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion
from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager
from core.domain.backtest.common import norm_code, precompute_limit_down, precompute_limit_up


logger = logging.getLogger(__name__)
SETS_PATH = ROOT / "core" / "domain" / "factors" / "factor_pool_state.json"
DEFAULT_OUT = ROOT / "data" / "experiments" / "expert_router_oos_h10.json"
LABEL_HORIZON = 10
LABEL_MODE = "tradable_open_to_open"
REBALANCE_STEP = 10
TOP_NS = (5, 10)
ROUTER_TOP_N = 10
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0001
STAMP_DUTY = 0.0005
BUY_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE
SELL_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE + STAMP_DUTY

WINDOWS = {
    "w1": {"train": ("2019-07-01", "2022-12-31"), "valid": ("2023-01-01", "2023-06-30"),
           "test": ("2023-07-01", "2024-06-30")},
    "w2": {"train": ("2019-07-01", "2023-12-31"), "valid": ("2024-01-01", "2024-06-30"),
           "test": ("2024-07-01", "2025-06-30")},
    "w3": {"train": ("2020-07-01", "2024-12-31"), "valid": ("2025-01-01", "2025-06-30"),
           "test": ("2025-07-01", "2026-06-30")},
}
FINAL_OOS = {
    "train": ("2021-07-01", "2025-12-31"),
    "valid": ("2026-01-01", "2026-06-30"),
    "test": ("2026-07-01", "2027-06-30"),
}

ROUTER_FEATURES = (
    "future_vol_probability",
    "trend_strength",
    "oversold_degree",
    "market_breadth",
    "sector_diffusion_proxy",
    "etf_flow_z",
    "margin_flow_z",
    "expert_disagreement",
    "market_data_missing",
    "sector_diffusion_missing",
    "etf_flow_missing",
    "margin_flow_missing",
)
VOL_MODEL_FEATURES = (
    "realized_vol_20",
    "realized_vol_60",
    "realized_vol_percentile",
    "trend_strength",
    "oversold_degree",
    "market_breadth",
    "dispersion",
)


def validate_windows(windows, order):
    previous_test_end = None
    for window_id in order:
        period = windows[window_id]
        train_end = pd.Timestamp(period["train"][1])
        valid_start, valid_end = map(pd.Timestamp, period["valid"])
        test_start, test_end = map(pd.Timestamp, period["test"])
        if not train_end < valid_start <= valid_end < test_start <= test_end:
            raise ValueError(f"窗口 {window_id} 的 train/valid/test 时间顺序无效")
        if previous_test_end is not None and test_start <= previous_test_end:
            raise ValueError(f"窗口 {window_id} 的测试期与前一窗口重叠")
        previous_test_end = test_end


validate_windows(WINDOWS, list(WINDOWS))


def load_factor_groups(path=SETS_PATH):
    curated = json.loads(path.read_text(encoding="utf-8"))["curated"]
    core = list(curated["core_base"]["flat_list"])
    return {
        "core": core,
        "trend": list(dict.fromkeys(core + curated["trend_satellite"]["flat_list"])),
        "rebound": list(dict.fromkeys(core + curated["rebound_satellite"]["flat_list"])),
    }


def _load_research_instruments(prefixes):
    from core.shared import config
    path = Path(config.QLIB_DATA_DIR) / "instruments" / "a500.txt"
    if path.exists():
        allowed = tuple(str(prefix) for prefix in prefixes)
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("\t", 1)[0].strip().lower()
            numeric = norm_code(code)
            if code and numeric.startswith(allowed):
                values.append(code)
        values = list(dict.fromkeys(values))
        if len(values) >= 1000:
            logger.info("使用Qlib历史股票池并集: %d只", len(values))
            return values
    values = build_filtered_instruments("a500", list(prefixes))
    if not isinstance(values, list):
        raise RuntimeError("无法构建研究股票池")
    return values


def _data_snapshot():
    snapshot = {}
    try:
        from core.data.connection import Database
        for table in ("daily_quotes", "etf_fund_flow", "margin_daily"):
            rows = Database.fetchall(f"SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date FROM {table}")
            snapshot[table] = rows[0] if rows else None
    except Exception as exc:
        snapshot["mysql_error"] = str(exc)
    try:
        from core.shared import config
        qlib_dir = Path(config.QLIB_DATA_DIR)
        for relative in (Path("calendars/day.txt"), Path("instruments/a500.txt")):
            path = qlib_dir / relative
            if path.exists():
                snapshot[str(relative).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as exc:
        snapshot["qlib_error"] = str(exc)
    return snapshot


def _implementation_hashes():
    paths = (
        Path(__file__),
        ROOT / "core" / "adapters" / "qlib" / "lstm_fusion.py",
        ROOT / "core" / "domain" / "backtest" / "common.py",
    )
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _stable_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _market_rows(start, end):
    try:
        from core.data.connection import Database
        query = """SELECT trade_date,
                          COUNT(*) AS n,
                          AVG(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) AS breadth,
                          AVG(pct_chg) AS mean_ret,
                          STDDEV_POP(pct_chg) AS dispersion,
                          SUM(amount) AS amount,
                          SUM(volume) AS volume
                     FROM daily_quotes
                    WHERE trade_date BETWEEN %s AND %s
                      AND pct_chg IS NOT NULL
                    GROUP BY trade_date ORDER BY trade_date"""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        rows = []
        chunk_start = start_ts
        while chunk_start <= end_ts:
            chunk_end = min(chunk_start + pd.Timedelta(days=180), end_ts)
            rows.extend(Database.fetchall(query, (chunk_start.strftime("%Y-%m-%d"),
                                                  chunk_end.strftime("%Y-%m-%d"))))
            chunk_start = chunk_end + pd.Timedelta(days=1)
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("市场截面聚合读取失败: %s", exc)
        return pd.DataFrame()


def _flow_rows(table, start, end, value_expr):
    try:
        from core.data.connection import Database
        return pd.DataFrame(Database.fetchall(
            f"SELECT trade_date, {value_expr} AS flow FROM {table} "
            "WHERE trade_date BETWEEN %s AND %s GROUP BY trade_date ORDER BY trade_date",
            (start, end),
        ))
    except Exception as exc:
        logger.warning("%s 读取失败，保留缺失标记: %s", table, exc)
        return pd.DataFrame()


def build_market_features(start, end, allow_missing=False):
    frame = _market_rows(start, end)
    if frame.empty:
        if not allow_missing:
            raise RuntimeError("daily_quotes 市场特征为空，正式实验禁止使用常数回退")
        dates = pd.date_range(start, end, freq="B")
        frame = pd.DataFrame({"trade_date": dates, "n": 0, "breadth": 0.5, "mean_ret": 0.0,
                              "dispersion": 0.0, "amount": 0.0, "volume": 0.0})
        market_missing = 1
    else:
        market_missing = 0
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    for col in ("breadth", "mean_ret", "dispersion", "amount", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["market_data_missing"] = frame[["breadth", "mean_ret", "dispersion"]].isna().any(axis=1).astype(int)
    if market_missing:
        frame["market_data_missing"] = 1
    frame[["mean_ret", "dispersion", "amount", "volume"]] = frame[
        ["mean_ret", "dispersion", "amount", "volume"]].fillna(0.0)
    frame["breadth"] = frame["breadth"].fillna(0.5)

    frame["market_ret_1d"] = frame["mean_ret"] / 100.0
    frame["market_level"] = (1.0 + frame["market_ret_1d"]).cumprod()
    ret = frame["market_level"].pct_change(fill_method=None)
    frame["realized_vol_20"] = ret.rolling(20, min_periods=10).std().fillna(0.0)
    frame["realized_vol_60"] = ret.rolling(60, min_periods=20).std().fillna(0.0)
    frame["realized_vol_percentile"] = frame["realized_vol_20"].rolling(252, min_periods=30).apply(
        lambda values: float(np.mean(values <= values[-1])), raw=True).fillna(0.5)
    ma20 = frame["market_level"].rolling(20, min_periods=5).mean()
    ma60 = frame["market_level"].rolling(60, min_periods=20).mean()
    frame["trend_strength"] = ((ma20 / ma60.replace(0, np.nan)) - 1.0).fillna(0.0)
    frame["oversold_degree"] = (-frame["market_level"].pct_change(20, fill_method=None)).clip(-1, 1).fillna(0.0)
    frame["market_breadth"] = frame["breadth"].clip(0, 1)
    frame["sector_diffusion_proxy"] = (
        frame["market_breadth"] * (1.0 - (frame["dispersion"].abs() / 100.0).clip(0, 1))
    ).fillna(0.0)
    frame["sector_diffusion_missing"] = 1

    etf = _flow_rows("etf_fund_flow", start, end, "SUM(COALESCE(net_inflow, 0))")
    margin = _flow_rows("margin_daily", start, end, "SUM(COALESCE(margin_buy, 0) - COALESCE(margin_repay, 0))")
    for name, flow in (("etf_flow", etf), ("margin_flow", margin)):
        if flow.empty:
            raw = pd.Series(np.nan, index=frame.index, dtype=float)
        else:
            flow["trade_date"] = pd.to_datetime(flow["trade_date"])
            raw = pd.to_numeric(flow.drop_duplicates("trade_date").set_index("trade_date")["flow"], errors="coerce")
            raw = raw.reindex(frame.index)
        raw = raw.shift(1)
        frame[f"{name}_missing"] = raw.isna().astype(int)
        raw = raw.fillna(0.0)
        mean = raw.rolling(60, min_periods=10).mean()
        scale = raw.rolling(60, min_periods=10).std().replace(0, np.nan)
        frame[f"{name}_z"] = ((raw - mean) / scale).clip(-5, 5).fillna(0.0)
    return frame.reset_index().rename(columns={"trade_date": "date"})


def _fit_future_vol_probability(market, train_period, allow_fallback=False):
    frame = market.sort_values("date").copy()
    dates = pd.to_datetime(frame["date"])
    future_vol = frame["market_ret_1d"].shift(-1).iloc[::-1].rolling(
        LABEL_HORIZON, min_periods=LABEL_HORIZON).std().iloc[::-1]
    matured_at = dates.shift(-LABEL_HORIZON)
    mask = ((dates >= pd.Timestamp(train_period[0])) & (dates <= pd.Timestamp(train_period[1])) &
            (matured_at <= pd.Timestamp(train_period[1])) & future_vol.notna())
    x = frame.loc[mask, list(VOL_MODEL_FEATURES)].fillna(0.0).to_numpy(dtype=float)
    if len(x) < 100:
        if not allow_fallback:
            raise RuntimeError(f"未来波动概率训练样本不足: {len(x)}")
        frame["future_vol_probability"] = frame["realized_vol_percentile"]
        return frame, {"fallback": True, "n_train": int(len(x))}
    threshold = float(future_vol.loc[mask].median())
    y = (future_vol.loc[mask] > threshold).astype(int).to_numpy()
    if len(set(y)) < 2:
        raise RuntimeError("未来波动概率训练标签只有一个类别")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    model.fit(x, y)
    all_x = frame.loc[:, list(VOL_MODEL_FEATURES)].fillna(0.0).to_numpy(dtype=float)
    frame["future_vol_probability"] = model.predict_proba(all_x)[:, 1]
    return frame, {"fallback": False, "n_train": int(len(x)), "threshold": threshold,
                   "positive_rate": float(y.mean())}


def _state_training_profile(market, train_period, role, mode):
    if mode == "full":
        return None, None, {"mode": mode, "role": role}
    dates = pd.to_datetime(market["date"])
    frame = market.loc[(dates >= pd.Timestamp(train_period[0])) & (dates <= pd.Timestamp(train_period[1]))].copy()
    if frame.empty:
        raise RuntimeError(f"{role} 的状态训练样本为空")
    affinity, _ = compute_state_affinity(frame, role)
    date_strings = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if mode == "weighted":
        values = 0.5 + 1.5 * affinity
        values = values / values.mean()
        return None, dict(zip(date_strings, values.astype(float))), {
            "mode": mode, "role": role, "n_dates": int(len(frame)), "weight_min": float(values.min()),
            "weight_max": float(values.max()), "weight_mean": float(values.mean()),
        }
    threshold = float(affinity.quantile(0.5))
    selected = date_strings.loc[affinity >= threshold].tolist()
    if len(selected) < 20:
        raise RuntimeError(f"{role} 状态子样本不足: {len(selected)}")
    return selected, None, {"mode": mode, "role": role, "n_dates": int(len(selected)),
                            "affinity_threshold": threshold}


def _router_features(market, date, disagreement=0.0):
    row = market.loc[pd.to_datetime(market["date"]) == pd.Timestamp(date)]
    defaults = {key: 0.0 for key in ROUTER_FEATURES}
    defaults.update({"future_vol_probability": 0.5, "market_breadth": 0.5,
                     "market_data_missing": 1.0, "sector_diffusion_missing": 1.0,
                     "etf_flow_missing": 1.0, "margin_flow_missing": 1.0,
                     "expert_disagreement": disagreement})
    if row.empty:
        return defaults
    source = row.iloc[0]
    return {key: disagreement if key == "expert_disagreement" else _safe_float(source.get(key), defaults[key])
            for key in ROUTER_FEATURES}


def _matured_history(history, cutoff):
    cutoff_ts = pd.Timestamp(cutoff)
    return sorted(
        [row for row in history if pd.Timestamp(row["label_matured_at"]) <= cutoff_ts],
        key=lambda row: (row["date"], row["label_matured_at"]),
    )


@dataclass
class _RouterModel:
    base: object
    calibrator: object
    training_rows: int
    calibrated: bool

    def predict_proba(self, row):
        x = np.asarray([[float(row.get(key, 0.0)) for key in ROUTER_FEATURES]], dtype=float)
        probability = float(self.base.predict_proba(x)[0, 1])
        if self.calibrator is not None:
            logit = math.log(np.clip(probability, 1e-6, 1 - 1e-6) /
                             np.clip(1 - probability, 1e-6, 1 - 1e-6))
            probability = float(self.calibrator.predict_proba([[logit]])[0, 1])
        return float(np.clip(probability, 0.05, 0.95))


def _fit_router(history, cutoff):
    rows = _matured_history(history, cutoff)
    if len(rows) < 20 or len({int(row["target"]) for row in rows}) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([[float(row.get(key, 0.0)) for key in ROUTER_FEATURES] for row in rows], dtype=float)
    y = np.asarray([int(row["target"]) for row in rows], dtype=int)
    magnitude = np.asarray([abs(float(row.get("advantage", 0.0))) for row in rows], dtype=float)
    positive = magnitude[magnitude > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    sample_weight = np.clip(magnitude / max(scale, 1e-6), 0.5, 3.0)

    def factory():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))

    calibrator = None
    if len(rows) >= 60:
        probabilities = []
        labels = []
        starts = [int(len(rows) * 0.5), int(len(rows) * 0.67), int(len(rows) * 0.84)]
        ends = starts[1:] + [len(rows)]
        for start, end in zip(starts, ends):
            if start < 20 or end <= start or len(set(y[:start])) < 2:
                continue
            fold = factory()
            fold.fit(x[:start], y[:start], logisticregression__sample_weight=sample_weight[:start])
            probabilities.extend(fold.predict_proba(x[start:end])[:, 1].tolist())
            labels.extend(y[start:end].tolist())
        if len(probabilities) >= 20 and len(set(labels)) == 2:
            p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            calibrator = LogisticRegression(C=1.0, max_iter=300).fit(logits, labels)
    base = factory()
    base.fit(x, y, logisticregression__sample_weight=sample_weight)
    return _RouterModel(base, calibrator, len(rows), calibrator is not None)


def _router_diagnostics(rows):
    valid = [row for row in rows if row.get("target") is not None and row.get("rebound_probability") is not None]
    if not valid:
        return {"n": 0, "auc": None, "brier": None, "log_loss": None, "ece": None}
    y = np.asarray([int(row["target"]) for row in valid])
    p = np.clip(np.asarray([float(row["rebound_probability"]) for row in valid]), 1e-6, 1 - 1e-6)
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else None
    ece = 0.0
    for left in np.linspace(0, 1, 6)[:-1]:
        right = left + 0.2
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return {"n": int(len(y)), "auc": auc, "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "ece": float(ece),
            "positive_rate": float(y.mean()), "mean_probability": float(p.mean())}


def _rank_fusion(trend, rebound, probability):
    joined = pd.concat([trend.rename("trend"), rebound.rename("rebound")], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    for col in ("trend", "rebound"):
        joined[col] = joined[col].groupby(level="datetime").rank(pct=True)
    return ((1.0 - probability) * joined["trend"] + probability * joined["rebound"]).rename("prediction")


def _hard_vol_switch(trend, rebound, market):
    joined = pd.concat([trend.rename("trend"), rebound.rename("rebound")], axis=1).dropna()
    output = []
    for date, group in joined.groupby(level="datetime"):
        row = market.loc[pd.to_datetime(market["date"]) == pd.Timestamp(date)]
        high_vol = bool(not row.empty and _safe_float(row.iloc[0].get("future_vol_probability"), 0.5) >= 0.5)
        output.append((group["rebound"] if high_vol else group["trend"]).rename("prediction"))
    return pd.concat(output).sort_index() if output else pd.Series(dtype=float)


def _expert_disagreement(trend, rebound, date):
    joined = pd.concat([trend.rename("trend"), rebound.rename("rebound")], axis=1).dropna()
    try:
        group = joined.xs(pd.Timestamp(date), level="datetime")
    except KeyError:
        return 0.0
    if group.empty:
        return 0.0
    return _safe_float((group["trend"].rank(pct=True) - group["rebound"].rank(pct=True)).abs().mean())


def _load_execution_constraints(start, end, instruments):
    from core.data.connection import Database
    codes = sorted({norm_code(code) for code in instruments})
    query_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    query_end = (pd.Timestamp(end) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    rows = []
    for offset in range(0, len(codes), 500):
        batch = codes[offset:offset + 500]
        placeholders = ",".join(["%s"] * len(batch))
        rows.extend(Database.fetchall(
            f"SELECT code, trade_date, open, close, adj_factor FROM daily_quotes "
            f"WHERE code IN ({placeholders}) AND trade_date BETWEEN %s AND %s ORDER BY code, trade_date",
            tuple(batch) + (query_start, query_end),
        ))
    if not rows:
        raise RuntimeError("成交约束数据为空，正式实验禁止忽略涨跌停")
    price_cache = {}
    frame = pd.DataFrame(rows)
    for code, group in frame.groupby("code"):
        price_cache[str(code)] = group.copy()
    coverage = len(price_cache) / max(len(codes), 1)
    if coverage < 0.95:
        raise RuntimeError(f"成交约束股票覆盖率不足: {coverage:.2%}")
    try:
        from core.data.fetcher import DataFetcher
        industry = DataFetcher().get_industry_map() or {}
        st_codes = {str(code) for code, info in industry.items()
                    if isinstance(info, dict) and "ST" in str(info.get("name", "")).upper()}
    except Exception:
        st_codes = set()
    up_cache, down_cache = {}, {}
    precompute_limit_up(price_cache, up_cache, st_codes)
    precompute_limit_down(price_cache, down_cache, st_codes)
    limit_up = {(pd.Timestamp(date).strftime("%Y-%m-%d"), code) for code, dates in up_cache.items() for date in dates}
    limit_down = {(pd.Timestamp(date).strftime("%Y-%m-%d"), code) for code, dates in down_cache.items() for date in dates}
    calendar = sorted(pd.to_datetime(frame["trade_date"].drop_duplicates()).tolist())
    return {"calendar": calendar, "limit_up": limit_up, "limit_down": limit_down,
            "coverage": coverage, "price_codes": len(price_cache)}


def _execution_date_maps(decision_dates, calendar):
    values = pd.DatetimeIndex(calendar)
    execution, maturity = {}, {}
    for value in sorted(pd.to_datetime(list(decision_dates))):
        position = int(values.searchsorted(value))
        if position >= len(values) or values[position] != value:
            continue
        if position + LABEL_HORIZON + 1 >= len(values):
            continue
        key = str(value.date())
        execution[key] = str(values[position + 1].date())
        maturity[key] = str(values[position + LABEL_HORIZON + 1].date())
    return execution, maturity


def _simulate_topn_portfolio(
    prediction, label, top_n, rebalance_step=REBALANCE_STEP,
    execution_dates=None, maturity_dates=None, limit_up=None, limit_down=None,
    buy_cost_rate=BUY_COST_RATE, sell_cost_rate=SELL_COST_RATE, decision_dates=None,
):
    frame = pd.concat([prediction.rename("prediction"), label.rename("label")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    execution_dates = execution_dates or {}
    maturity_dates = maturity_dates or {}
    limit_up = limit_up or set()
    limit_down = limit_down or set()
    available_dates = sorted(set(pd.to_datetime(frame.index.get_level_values("datetime"))))
    if decision_dates is None:
        decision_dates = available_dates[::max(int(rebalance_step), 1)]
    else:
        decision_dates = sorted(set(pd.to_datetime(decision_dates)))
        missing_dates = set(decision_dates) - set(available_dates)
        if missing_dates:
            raise ValueError(f"决策日缺少有效预测/标签: {sorted(missing_dates)[:5]}")
    current = {}
    equity = 1.0
    equity_curve = [1.0]
    periods = []
    traded_notional = 0.0
    one_way_turnover = 0.0
    for date in decision_dates:
        date_str = str(pd.Timestamp(date).date())
        exec_date = execution_dates.get(date_str)
        matured_at = maturity_dates.get(date_str)
        if not exec_date or not matured_at:
            continue
        group = frame.xs(date, level="datetime").sort_values("prediction", ascending=False)
        desired = [str(code) for code in group.head(top_n).index]
        ideal = {code: 1.0 / top_n for code in desired}
        target = dict(current)
        locked = set()
        for code, weight in list(current.items()):
            desired_weight = ideal.get(code, 0.0)
            if desired_weight < weight and (exec_date, norm_code(code)) in limit_down:
                locked.add(code)
                target[code] = weight
            else:
                target[code] = desired_weight
        for code, desired_weight in ideal.items():
            current_weight = current.get(code, 0.0)
            if desired_weight > current_weight and (exec_date, norm_code(code)) in limit_up:
                target[code] = current_weight
                locked.add(code)
            elif code not in target:
                target[code] = desired_weight
        target = {code: float(weight) for code, weight in target.items() if weight > 1e-12}
        total = sum(target.values())
        if total > 1.0 + 1e-12:
            locked_total = sum(target.get(code, 0.0) for code in locked)
            flexible = [code for code in target if code not in locked]
            flexible_total = sum(target[code] for code in flexible)
            remaining = max(1.0 - locked_total, 0.0)
            scale = remaining / flexible_total if flexible_total > 0 else 0.0
            for code in flexible:
                target[code] *= scale
        sells = sum(max(current.get(code, 0.0) - target.get(code, 0.0), 0.0)
                    for code in set(current) | set(target))
        buys = sum(max(target.get(code, 0.0) - current.get(code, 0.0), 0.0)
                   for code in set(current) | set(target))
        cost = buys * buy_cost_rate + sells * sell_cost_rate
        returns = {str(code): float(value) for code, value in group["label"].items()}
        gross = sum(weight * returns.get(code, 0.0) for code, weight in target.items())
        cash = max(1.0 - sum(target.values()), 0.0)
        end_values = {code: weight * (1.0 + returns.get(code, 0.0)) for code, weight in target.items()}
        pre_cost_total = cash + sum(end_values.values())
        end_total = pre_cost_total * (1.0 - cost)
        if end_total <= 0:
            raise RuntimeError(f"{date_str} 组合权益异常")
        net = end_total - 1.0
        equity *= end_total
        equity_curve.append(equity)
        current = {code: value / pre_cost_total for code, value in end_values.items() if value > 0}
        traded_notional += buys + sells
        one_way_turnover += 0.5 * (buys + sells)
        periods.append({
            "date": date_str, "execution_date": exec_date, "label_matured_at": matured_at,
            "gross": gross, "net": net, "cost": cost, "buy_notional": buys, "sell_notional": sells,
            "turnover": 0.5 * (buys + sells), "universe": float(group["label"].mean()),
            "holdings": sorted(target),
        })
    ic = frame.groupby(level="datetime", sort=True).apply(
        lambda group: group["prediction"].corr(group["label"], method="spearman")
        if group["prediction"].nunique() > 1 and group["label"].nunique() > 1 else np.nan,
        include_groups=False,
    ).dropna()
    returns = np.asarray([row["net"] for row in periods], dtype=float)
    annual_factor = 252 / max(rebalance_step, 1)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    peak = np.maximum.accumulate(np.asarray(equity_curve, dtype=float))
    drawdown = np.asarray(equity_curve) / np.where(peak > 0, peak, 1.0) - 1.0
    return {
        "top_n": int(top_n), "n_periods": int(len(periods)),
        "mean_net": float(returns.mean()) if len(returns) else None,
        "total_return": float(equity - 1.0),
        "annual_return": float(equity ** (annual_factor / len(returns)) - 1.0) if len(returns) else None,
        "sharpe": float(returns.mean() / std * math.sqrt(annual_factor)) if std > 0 else None,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "positive_ratio": float((returns > 0).mean()) if len(returns) else None,
        "turnover": float(one_way_turnover / len(periods)) if periods else None,
        "traded_notional": float(traded_notional),
        "transaction_cost": float(sum(row["cost"] for row in periods)),
        "rank_ic": float(ic.mean()) if len(ic) else None,
        "icir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else None,
        "periods": periods,
    }


def _evaluate_prediction(prediction, label, execution, maturity, constraints,
                         rebalance_step=REBALANCE_STEP, decision_dates=None):
    results = {}
    for top_n in TOP_NS:
        results[f"top{top_n}"] = _simulate_topn_portfolio(
            prediction, label, top_n, rebalance_step, execution, maturity,
            constraints["limit_up"], constraints["limit_down"],
            decision_dates=decision_dates,
        )
    return results


def _public_metrics(results):
    return {key: {name: value for name, value in metrics.items() if name != "periods"}
            for key, metrics in results.items()}


def _load_previous(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="冻结因子 + 专家训练方式 + 市场路由严格 walk-forward OOS")
    parser.add_argument("--modes", nargs="+", choices=("full", "weighted", "subset"), default=["full"])
    parser.add_argument("--windows", nargs="+", choices=tuple(WINDOWS), default=["w1"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="清空本实验输出后从第一窗口重跑")
    parser.add_argument("--allow-missing-market", action="store_true", help="仅冒烟使用，正式实验禁止")
    parser.add_argument("--run-final-oos", action="store_true")
    parser.add_argument("--final-mode", choices=("full", "weighted", "subset"))
    args = parser.parse_args()

    requested_modes = list(dict.fromkeys(args.modes))
    requested_windows = list(dict.fromkeys(args.windows))
    window_positions = [list(WINDOWS).index(window_id) for window_id in requested_windows]
    if window_positions != sorted(window_positions):
        raise ValueError("--windows 必须按时间顺序传入")
    validate_windows(WINDOWS, requested_windows)
    if args.run_final_oos and not args.final_mode:
        raise ValueError("运行最终OOS必须显式指定 --final-mode")

    previous = _load_previous(args.output)
    if args.force:
        if previous and previous.get("final_oos_lock"):
            raise RuntimeError("最终OOS已经锁定，禁止 --force 覆盖")
        previous = None
    if previous:
        groups = {key: list(value) for key, value in previous["config"]["factor_groups"].items()}
        instruments = list(previous["config"]["instruments"])
    else:
        groups = load_factor_groups()
        prefixes = ["600", "601", "603", "605", "000", "001", "002", "003"]
        instruments = _load_research_instruments(prefixes)
    data_snapshot = _data_snapshot()
    fingerprint_payload = {
        "schema_version": 2, "label_horizon": LABEL_HORIZON, "label_mode": LABEL_MODE,
        "rebalance_step": REBALANCE_STEP, "top_ns": TOP_NS, "router_top_n": ROUTER_TOP_N,
        "costs": {"commission": COMMISSION_RATE, "slippage": SLIPPAGE_RATE, "stamp_duty": STAMP_DUTY},
        "windows": WINDOWS, "final_oos": FINAL_OOS, "modes": requested_modes, "epochs": args.epochs,
        "factor_groups": groups, "instruments": instruments,
        "factor_hash": _stable_hash(groups), "instrument_hash": _stable_hash(sorted(instruments)),
        "router_features": ROUTER_FEATURES, "implementation_hashes": _implementation_hashes(),
    }
    fingerprint = _stable_hash(fingerprint_payload)
    if previous and previous.get("experiment_fingerprint") != fingerprint:
        raise RuntimeError("输出文件的实验指纹不一致，请使用新的 --output；不要混合不同配置")
    output = previous or {
        "schema_version": 2, "experiment_fingerprint": fingerprint, "config": fingerprint_payload,
        "factor_source": str(SETS_PATH), "instrument_count": len(instruments), "windows": {},
        "development_data_snapshot": data_snapshot,
        "router_history": {mode: [] for mode in requested_modes},
    }
    if args.run_final_oos:
        if output.get("final_oos_lock"):
            raise RuntimeError("最终OOS已评估，禁止重复查看")
        missing = [window_id for window_id in WINDOWS if window_id not in output["windows"]]
        if missing:
            raise RuntimeError(f"最终OOS前必须完成全部开发窗口: {missing}")
        run_windows = [("final", FINAL_OOS)]
        run_modes = [args.final_mode]
    else:
        for window_id in requested_windows:
            prior = list(WINDOWS)[:list(WINDOWS).index(window_id)]
            missing = [item for item in prior if item not in output["windows"] and item not in requested_windows]
            if missing:
                raise RuntimeError(f"必须先完成更早的开发窗口: {missing}")
        run_windows = [(window_id, WINDOWS[window_id]) for window_id in requested_windows]
        run_modes = requested_modes

    init_qlib()
    manager = QlibModelManager()
    union = list(dict.fromkeys(key for keys in groups.values() for key in keys))
    feature_start = (pd.Timestamp(min(period["train"][0] for _, period in run_windows)) -
                     pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    feature_end = max(period["test"][1] for _, period in run_windows)
    print(f"loading causal market features {feature_start} -> {feature_end}", flush=True)
    market_base = build_market_features(feature_start, feature_end, args.allow_missing_market)
    print(f"market feature dates={len(market_base)}", flush=True)

    for window_id, period in run_windows:
        if window_id != "final" and window_id in output["windows"]:
            print(f"skip completed window={window_id}", flush=True)
            continue
        print(f"window={window_id} train={period['train']} valid={period['valid']} test={period['test']}", flush=True)
        market, vol_info = _fit_future_vol_probability(market_base, period["train"], args.allow_missing_market)
        source_config = manager._build_task_config(
            "lightgbm_rank", "WhiteBoxHandler", instruments,
            period["train"], period["valid"], period["test"],
            model_params=dict(PARAM_PROFILES["conservative"]["lightgbm"]),
            label_horizon=LABEL_HORIZON, custom_factor_keys=union, label_norm="csrank",
            enable_cross_sectional=False, enable_winsorize=True, label_mode=LABEL_MODE,
        )
        from qlib.utils import init_instance_by_config
        source = init_instance_by_config(source_config["dataset"])
        from qlib.data.dataset.handler import DataHandlerLP
        raw_label = source.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
        if hasattr(raw_label, "columns"):
            raw_label = raw_label.iloc[:, 0]
        if raw_label.empty:
            raise RuntimeError(f"{window_id} 测试标签为空；最终OOS数据可能尚未成熟")
        print(f"loading execution constraints window={window_id}", flush=True)
        constraints = _load_execution_constraints(period["test"][0], period["test"][1], instruments)
        print(f"execution coverage={constraints['coverage']:.2%} codes={constraints['price_codes']}", flush=True)
        decision_dates = sorted(set(pd.to_datetime(raw_label.index.get_level_values("datetime"))))
        execution, maturity = _execution_date_maps(decision_dates, constraints["calendar"])
        if len(execution) < len(decision_dates) * 0.9:
            raise RuntimeError(f"{window_id} 成交日/标签成熟日映射覆盖不足")
        portfolio_dates = [
            date for date in decision_dates[::REBALANCE_STEP]
            if str(date.date()) in execution and str(date.date()) in maturity
        ]
        window_result = {"period": period, "modes": {}, "router": {}, "volatility_model": vol_info,
                         "data_snapshot": data_snapshot,
                         "execution_constraints": {"coverage": constraints["coverage"],
                                                   "price_codes": constraints["price_codes"]}}

        core_result = train_lgb_lstm_fusion(
            factor_keys=groups["core"], instruments=instruments,
            train_period=period["train"], valid_period=period["valid"], test_period=period["test"],
            label_horizon=LABEL_HORIZON, label_mode=LABEL_MODE, label_norm="csrank",
            fusion_weight=0.5, param_profile="conservative", lstm_epochs=args.epochs, dataset=source,
        )
        core_prediction = core_result["fused_pred"].reindex(raw_label.index)
        core_portfolios = _evaluate_prediction(
            core_prediction, raw_label, execution, maturity, constraints,
            decision_dates=portfolio_dates)
        window_result["core_baseline"] = _public_metrics(core_portfolios)
        del core_result
        gc.collect()

        expert_outputs = {}
        for mode in run_modes:
            mode_result = {"core": window_result["core_baseline"]}
            expert_outputs[mode] = {}
            for role in ("trend", "rebound"):
                train_dates, train_weights, profile = _state_training_profile(
                    market, period["train"], role, mode)
                print(f"training window={window_id} mode={mode} role={role} factors={len(groups[role])} ",
                      f"profile_dates={profile.get('n_dates', 'all')}", flush=True)
                result = train_lgb_lstm_fusion(
                    factor_keys=groups[role], instruments=instruments,
                    train_period=period["train"], valid_period=period["valid"], test_period=period["test"],
                    label_horizon=LABEL_HORIZON, label_mode=LABEL_MODE, label_norm="csrank",
                    fusion_weight=0.5, param_profile="conservative", lstm_epochs=args.epochs,
                    dataset=source, train_dates=train_dates, train_weights=train_weights,
                )
                prediction = result["fused_pred"].reindex(raw_label.index)
                portfolios = _evaluate_prediction(
                    prediction, raw_label, execution, maturity, constraints,
                    decision_dates=portfolio_dates)
                mode_result[role] = {"training_profile": profile, "metrics": _public_metrics(portfolios)}
                expert_outputs[mode][role] = {"prediction": prediction, "portfolios": portfolios}
                del result
                gc.collect()
            window_result["modes"][mode] = mode_result

        for mode in run_modes:
            experts = expert_outputs[mode]
            trend = experts["trend"]["prediction"]
            rebound = experts["rebound"]["prediction"]
            trend_periods = {row["date"]: row for row in experts["trend"]["portfolios"][f"top{ROUTER_TOP_N}"]["periods"]}
            rebound_periods = {row["date"]: row for row in experts["rebound"]["portfolios"][f"top{ROUTER_TOP_N}"]["periods"]}
            history = output["router_history"].setdefault(mode, [])
            rows = []
            soft_parts = []
            for date in sorted(set(trend_periods) & set(rebound_periods)):
                trend_row = trend_periods[date]
                rebound_row = rebound_periods[date]
                disagreement = _expert_disagreement(trend, rebound, date)
                features = _router_features(market, date, disagreement)
                router = _fit_router(history, date)
                probability = router.predict_proba(features) if router else 0.5
                advantage = float(rebound_row["net"] - trend_row["net"])
                target = int(advantage > 0)
                history_row = {
                    **features, "target": target, "date": date,
                    "label_matured_at": max(trend_row["label_matured_at"], rebound_row["label_matured_at"]),
                    "advantage": advantage, "top_n": ROUTER_TOP_N,
                }
                history.append(history_row)
                rows.append({
                    **history_row, "rebound_probability": probability,
                    "router_training_rows": router.training_rows if router else 0,
                    "router_calibrated": bool(router and router.calibrated),
                    "trend_net": trend_row["net"], "rebound_net": rebound_row["net"],
                })
                try:
                    t_group = trend.xs(pd.Timestamp(date), level="datetime", drop_level=False)
                    r_group = rebound.xs(pd.Timestamp(date), level="datetime", drop_level=False)
                    soft_parts.append(_rank_fusion(t_group, r_group, probability))
                except KeyError:
                    pass
            soft_prediction = pd.concat(soft_parts).sort_index() if soft_parts else pd.Series(dtype=float)
            soft_portfolios = _evaluate_prediction(
                soft_prediction, raw_label, execution, maturity, constraints,
                decision_dates=portfolio_dates)
            hard_prediction = _hard_vol_switch(trend, rebound, market)
            hard_portfolios = _evaluate_prediction(
                hard_prediction, raw_label, execution, maturity, constraints,
                decision_dates=portfolio_dates)
            window_result["router"][mode] = {
                "training_rows_before": len(_matured_history(history[:-len(rows)] if rows else history, period["test"][0])),
                "training_rows_after": len(history), "daily": rows,
                "diagnostics": _router_diagnostics(rows),
                "fixed_trend": _public_metrics(experts["trend"]["portfolios"]),
                "hard_vol_switch": _public_metrics(hard_portfolios),
                "soft_fusion": _public_metrics(soft_portfolios),
            }
            output["router_history"][mode] = history

        if window_id == "final":
            output["final_oos"] = window_result
            output["final_oos_lock"] = {
                "mode": args.final_mode, "period": FINAL_OOS["test"],
                "experiment_fingerprint": fingerprint, "result_hash": _stable_hash(window_result),
            }
        else:
            output["windows"][window_id] = window_result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        del source
        gc.collect()
        print(f"saved {args.output}", flush=True)
    print(json.dumps({"output": str(args.output), "fingerprint": fingerprint,
                      "completed_windows": list(output["windows"]),
                      "final_oos_locked": bool(output.get("final_oos_lock"))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
