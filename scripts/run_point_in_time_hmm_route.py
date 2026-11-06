import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ["REGIME_HMM_EXTRA_FEATURES"] = "true"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contextual_top5_weekly_corrected import (
    END,
    START,
    REBOUND_RUN,
    TREND_RUN,
    _account_constrained,
    _expert_weekly,
    _load_industry_map,
    _load_open,
    _load_prediction,
    _load_trade_data,
    _weekly_dates,
)
from core.domain.regime.classifier import HMMRegimeClassifier
from core.domain.timing.breadth import get_breadth_history
from core.data import Database

INPUT = ROOT / "data" / "experiments" / "contextual_top5_weekly_corrected_20260916.json"
OUT = ROOT / "data" / "experiments" / "hmm_point_in_time_route_20260916.json"
REPORT = ROOT / "docs" / "hmm_point_in_time_route_20260916.md"

def main():
    trend = _load_prediction(TREND_RUN)
    rebound = _load_prediction(REBOUND_RUN)
    common = sorted(set(trend.index.get_level_values("date")) & set(rebound.index.get_level_values("date")))
    common = [d for d in common if START <= d <= END]
    weekly = _weekly_dates(common)
    execution_map = {d: common[common.index(d) + 1] for d in weekly if d in common and common.index(d) + 1 < len(common)}
    decision_dates = [d for d in weekly if d in execution_map]
    next_exec = [execution_map[d] for d in decision_dates]
    all_codes = set(trend.index.get_level_values("code")) & set(rebound.index.get_level_values("code"))
    opens = _load_open(all_codes, next_exec + next_exec[1:])
    trade_data = _load_trade_data(all_codes, next_exec + next_exec[1:])
    industry_map = _load_industry_map()
    t_week, t_baskets = _expert_weekly(trend, opens, decision_dates, next_exec)
    r_week, r_baskets = _expert_weekly(rebound, opens, decision_dates, next_exec)
    merged = t_week.merge(r_week, on=["date", "execution", "following_execution"], suffixes=("_trend", "_rebound"))
    merged = merged.sort_values("date").reset_index(drop=True)
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
    route_rows = []
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
            state = "missing"
            confidence = 0.0
            route = "trend"
        else:
            result = classifier.predict(index_df)
            state = result.regime.value
            confidence = float(result.confidence)
            route = "trend" if state in ("volatile_up", "steady_up") else "rebound"
        route_rows.append({"date": str(cutoff.date()), "state": state, "confidence": confidence, "route": route})
    route_frame = pd.DataFrame(route_rows)
    merged["route_hmm_pit"] = route_frame["route"].to_numpy()
    merged["selected_net_hmm_pit"] = merged.apply(lambda r: r["net_rebound"] if r.route_hmm_pit == "rebound" else r.net_trend, axis=1)
    merged["route_g0"] = ["trend"] * len(merged)
    old = json.loads(INPUT.read_text(encoding="utf-8"))
    old_trace = pd.DataFrame(old["weekly_trace"])
    old_trace["date"] = pd.to_datetime(old_trace["date"])
    g0_map = dict(zip(old_trace.date, old_trace.route_G0_volatility))
    merged["route_g0"] = merged.date.map(g0_map).fillna("trend")
    merged["selected_net_g0"] = merged.apply(lambda r: r["net_rebound"] if r.route_g0 == "rebound" else r.net_trend, axis=1)
    routes = {"hmm_point_in_time": merged.route_hmm_pit.tolist(), "G0_volatility": merged.route_g0.tolist()}
    accounts = {}
    stress_accounts = {}
    for name, route in routes.items():
        baskets = {str(d.date()): (r_baskets if route[i] == "rebound" else t_baskets).get(str(d.date()), []) for i, d in enumerate(merged.date)}
        account = _account_constrained(merged.set_index("date"), baskets, trade_data, industry_map)
        accounts[name] = account
        stress_accounts[name] = _account_constrained(merged.set_index("date"), baskets, trade_data, industry_map, 0.0008, 0.0018)
    chosen = merged.selected_net_hmm_pit.to_numpy()
    result = {
        "status": "point_in_time_hmm_contract_diagnostic",
        "decision_weeks": int(len(merged)),
        "hmm_feature_mode": "production_extra_features_cached",
        "hmm_train_cutoff": "each decision date inclusive",
        "hmm_route_rebound_count": int(merged.route_hmm_pit.eq("rebound").sum()),
        "hmm_route_mean_net": float(chosen.mean()),
        "hmm_route_accuracy_vs_realized_best": float((merged.route_hmm_pit == merged.apply(lambda r: "rebound" if r.net_rebound > r.net_trend else "trend", axis=1)).mean()),
        "g0_mean_net": float(merged.selected_net_g0.mean()),
        "g0_route_rebound_count": int(merged.route_g0.eq("rebound").sum()),
        "accounts": accounts,
        "stress_accounts": stress_accounts,
        "state_distribution": route_frame.state.value_counts().to_dict(),
        "weekly_trace": pd.concat([merged, route_frame.drop(columns=["date"])], axis=1).to_dict(orient="records"),
        "limitations": [
            "宽度和北向额外特征通过一次历史缓存注入；HMM每4周更新一次，期间使用最近一次模型。",
            "状态到专家采用冻结规则：up选trend，其余选rebound；未在当前周收益上学习映射。",
            "只覆盖59个周，且账户仍继承本轮历史行业映射和Qlib回放限制。",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 点时 HMM 双专家路由诊断（2026-09-16）", "", "状态：`point_in_time_hmm_contract_diagnostic`。每个决策日只使用该日及以前指数数据；宽度和北向特征通过一次历史缓存注入。本报告仍是59周机制诊断，不是跨季度验收。", "", "| 项目 | 值 |", "|---|---:|", f"| 决策周 | {len(merged)} |", f"| HMM选择反弹 | {int(merged.route_hmm_pit.eq('rebound').sum())} |", f"| HMM周均净收益 | {chosen.mean():.5f} |", f"| HMM事后最佳专家比例 | {result['hmm_route_accuracy_vs_realized_best']:.3f} |", f"| G0周均净收益 | {merged.selected_net_g0.mean():.5f} |", f"| HMM约束账户终值 | {accounts['hmm_point_in_time']['final_nav']:.5f} |", f"| G0约束账户终值 | {accounts['G0_volatility']['final_nav']:.5f} |", "", "## 判定", "", "- 该版本解决了额外特征常数问题，并按历史截断数据完成点时HMM诊断。", "- 模型每4周更新一次，期间沿用最近一次拟合；状态到专家映射仍是冻结规则。", "- 只有至少5个季度的生产合同回放通过，才能判定HMM是否具有正式增量。", "", f"状态分布：`{result['state_distribution']}`。", f"原始 JSON：`{OUT.relative_to(ROOT)}`。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["decision_weeks", "hmm_route_rebound_count", "hmm_route_mean_net", "hmm_route_accuracy_vs_realized_best", "g0_mean_net", "accounts", "state_distribution"]}, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
