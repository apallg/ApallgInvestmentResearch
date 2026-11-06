import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contextual_top5_weekly_corrected import (
    _account_constrained, _expert_weekly, _load_industry_map, _load_open,
    _load_trade_data, _weekly_dates, _bootstrap, BUY_COST, SELL_COST,
)
from core.domain.regime.classifier import HMMRegimeClassifier
from core.domain.timing.breadth import get_breadth_history
from core.data import Database

ROLLING = ROOT / "data/experiments/formal_rolling_expert_oos_20260916.json"
OUT = ROOT / "data/experiments/formal_rolling_router_account_20260916.json"
REPORT = ROOT / "docs/formal_rolling_router_account_20260916.md"


def load_round_predictions(model):
    result = []
    payload = json.loads(ROLLING.read_text(encoding="utf-8"))[model]
    for row in payload["rounds"]:
        paths = list(ROOT.glob(f"mlruns/{row['experiment_id']}/{row['run_id']}/artifacts/pred.pkl"))
        if not paths:
            raise FileNotFoundError(f"missing pred artifact: {row['experiment_id']}/{row['run_id']}")
        obj = pickle.loads(paths[0].read_bytes())
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        obj.index = pd.MultiIndex.from_tuples(obj.index, names=["date", "code"])
        obj.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(obj.index.get_level_values(0)), obj.index.get_level_values(1)
        ], names=["date", "code"])
        test_start, test_end = map(pd.Timestamp, row["test"])
        result.append(obj[(obj.index.get_level_values("date") >= test_start) & (obj.index.get_level_values("date") <= test_end)])
    out = pd.concat(result).sort_index()
    return out[~out.index.duplicated(keep="last")].rename("score").astype(float)


def main():
    trend = load_round_predictions("trend")
    rebound = load_round_predictions("rebound")
    common = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    weekly = _weekly_dates(common)
    execution_map = {d: common[common.index(d) + 1] for d in weekly if d in common and common.index(d) + 1 < len(common)}
    decision_dates = [d for d in weekly if d in execution_map]
    next_exec = [execution_map[d] for d in decision_dates]
    codes = set(trend.index.get_level_values("code")) & set(rebound.index.get_level_values("code"))
    opens = _load_open(codes, next_exec + next_exec[1:])
    trade_data = _load_trade_data(codes, next_exec + next_exec[1:])
    industry_map = _load_industry_map()
    t_week, t_baskets = _expert_weekly(trend, opens, decision_dates, next_exec)
    r_week, r_baskets = _expert_weekly(rebound, opens, decision_dates, next_exec)
    merged = t_week.merge(r_week, on=["date", "execution", "following_execution"], suffixes=("_trend", "_rebound"))
    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("no common weekly expert results")

    index_df_all = __import__("core.data.fetcher", fromlist=["DataFetcher"]).DataFetcher().get_index_daily("沪深300", days=365 * 10)
    index_df_all["date"] = pd.to_datetime(index_df_all["date"])
    breadth = get_breadth_history(str(index_df_all.date.min().date()), str(index_df_all.date.max().date())).reset_index()
    breadth["trade_date"] = pd.to_datetime(breadth["trade_date"])
    north_rows = Database.fetchall(
        "SELECT trade_date, north_net FROM north_flow_summary WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date",
        (str(index_df_all.date.min().date()), str(index_df_all.date.max().date())),
    )
    north = pd.DataFrame(north_rows, columns=["trade_date", "north_net"]) if north_rows else pd.DataFrame(columns=["trade_date", "north_net"])
    if not north.empty:
        north["trade_date"] = pd.to_datetime(north["trade_date"])

    class CachedHMM(HMMRegimeClassifier):
        def _enrich_features(self, df):
            out = df.copy()
            out["trade_date"] = pd.to_datetime(out["date"] if "date" in out.columns else out["trade_date"])
            out = out.merge(breadth[["trade_date", "up_ratio", "pct_median"]], on="trade_date", how="left")
            if not north.empty:
                out = out.merge(north, on="trade_date", how="left")
            return out

    hmm_routes = []
    classifier = None
    for i, row in merged.iterrows():
        cutoff = pd.Timestamp(row["date"])
        index_df = index_df_all[index_df_all["date"] <= cutoff]
        if classifier is None or i % 4 == 0:
            classifier = CachedHMM()
            ok = classifier.fit(index_df, random_state=20260916)
        else:
            ok = classifier._trained
        if not ok:
            hmm_routes.append("trend")
        else:
            state = classifier.predict(index_df).regime.value
            hmm_routes.append("trend" if state in ("volatile_up", "steady_up") else "rebound")
    index_series = index_df_all.set_index("date")["close"].sort_index()
    index_ret = index_series.pct_change(fill_method=None)
    vol_series = index_ret.rolling(15, min_periods=15).std() * np.sqrt(252.0)
    context = pd.DataFrame({"vol15": vol_series, "dvol5": vol_series - vol_series.shift(5)})
    merged = merged.join(context, on="date")
    merged["label_matured_at"] = pd.to_datetime(merged["following_execution"])
    merged["advantage"] = merged["net_rebound"] - merged["net_trend"]
    vol_routes = ["trend"] * len(merged)
    for i in range(26, len(merged)):
        row = merged.iloc[i]
        prior = merged.iloc[:i]
        train = prior.dropna(subset=["vol15", "dvol5", "advantage"])
        if len(train) < 26 or row[["vol15", "dvol5"]].isna().any():
            continue
        model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=8, random_state=20260916)
        model.fit(train[["vol15", "dvol5"]].to_numpy(), train["advantage"].to_numpy())
        vol_routes[i] = "rebound" if float(model.predict(row[["vol15", "dvol5"]].to_numpy().reshape(1, -1))[0]) > 0.001 else "trend"
    routes = {
        "fixed_trend": ["trend"] * len(merged),
        "fixed_rebound": ["rebound"] * len(merged),
        "G0_volatility": vol_routes,
        "HMM_point_in_time": hmm_routes,
    }
    accounts = {}
    details = {}
    for name, route in routes.items():
        baskets = {str(d.date()): (r_baskets if route[i] == "rebound" else t_baskets).get(str(d.date()), []) for i, d in enumerate(merged.date)}
        frame = merged.set_index("date")
        account = _account_constrained(frame, baskets, trade_data, industry_map)
        accounts[name] = account
        values = np.array([row.net_rebound if route[i] == "rebound" else row.net_trend for i, row in merged.iterrows()], dtype=float)
        realized = np.array([row.net_rebound if row.net_rebound > row.net_trend else row.net_trend for _, row in merged.iterrows()])
        details[name] = {
            "weeks": len(values), "rebound_count": int(sum(x == "rebound" for x in route)),
            "mean_net": float(values.mean()), "mean_regret": float((realized - values).mean()),
            "accuracy_vs_best": float(np.mean([(route[i] == "rebound") == (merged.iloc[i].net_rebound > merged.iloc[i].net_trend) for i in range(len(merged))])),
            "ci_4week": _bootstrap(values, 4),
        }
    quarterly = {}
    for name, account in accounts.items():
        frame = pd.DataFrame(account["daily"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame["quarter"] = frame["date"].dt.to_period("Q").astype(str)
        quarterly[name] = []
        for quarter, group in frame.groupby("quarter", sort=True):
            nav = group["nav"].to_numpy(dtype=float)
            quarterly[name].append({"quarter": quarter, "weeks": int(len(group)), "return": float(nav[-1] / nav[0] - 1.0) if len(nav) > 1 else 0.0, "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1.0))})
    payload = {"status": "formal_rolling_same_contract_account", "decision_weeks": len(merged), "period": [str(merged.date.min().date()), str(merged.date.max().date())], "routes": details, "accounts": accounts, "quarterly": quarterly, "state_route": {"hmm": hmm_routes, "g0": vol_routes}}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 正式滚动专家四路账户回放（2026-09-16）", "", f"测试期：{payload['period'][0]} 至 {payload['period'][1]}，有效决策周 {len(merged)}。每个季度使用独立测试期 pred.pkl，账户统一使用 Top5、下一交易日开盘、95%目标仓位、单票/行业/换手/整手/涨跌停约束。", "", "| 方案 | 周数 | 选反弹 | 周均净收益 | 平均 regret | 最佳专家命中率 | 账户终值 | 最大回撤 | 平均换手 | 平均仓位 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in routes:
        d, a = details[name], accounts[name]
        lines.append(f"| {name} | {d['weeks']} | {d['rebound_count']} | {d['mean_net']:.5f} | {d['mean_regret']:.5f} | {d['accuracy_vs_best']:.3f} | {a['final_nav']:.5f} | {a['max_drawdown']:.5f} | {a['mean_turnover']:.3f} | {a['mean_exposure']:.3f} |")
    lines += ["", "## 季度收益与回撤", "", "| 季度 | 固定趋势 | 固定反弹 | G0波动率 | HMM |", "|---|---:|---:|---:|---:|"]
    quarters = [item["quarter"] for item in quarterly["fixed_trend"]]
    for quarter in quarters:
        vals = {name: next(item for item in quarterly[name] if item["quarter"] == quarter) for name in routes}
        lines.append(f"| {quarter} | {vals['fixed_trend']['return']:.2%} | {vals['fixed_rebound']['return']:.2%} | {vals['G0_volatility']['return']:.2%} | {vals['HMM_point_in_time']['return']:.2%} |")
    lines += ["", "## 判定", "", "- 该回放使用滚动训练产生的真实测试期预测，未使用单次全样本模型替代季度模型。", "- HMM 仅使用决策日前指数、宽度和北向数据，每4周重新拟合；状态映射固定为上涨状态选趋势、其余选反弹。", "- 账户终值、换手和仓位共同决定是否具有交易增量；训练 IC 不作为生产批准依据。", "", f"原始数据：`{OUT.relative_to(ROOT)}`。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
