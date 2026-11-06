import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.adapters.qlib.rolling_train import rolling_train_dual

OUT = ROOT / "data" / "experiments" / "formal_rolling_expert_oos_20260916.json"

TREND_KEYS = [
    "dp_historical_n12_delta_window5", "dp_historical_n8_delta_window5", "dp_historical_n4_delta_window5",
    "SP_n10", "SP_n60", "SP_n5", "SP_n120", "SP_n20", "senti", "fund_turnover", "a191_liq_04",
    "std_turn_6m", "a191_liq_11", "a101_a040", "a101_a018", "a191_mom_18", "a191_mom_19",
    "a191_vol_01", "a101_a032", "a101_a054", "a191_vol_25", "volume_variance_n10_delta_window5",
    "a191_vol_10", "a191_hof_04", "a101_a063", "a101_a090", "a101_a057", "a101_a015", "a101_a075",
    "a101_a087", "a101_a059", "a101_a082", "a158_roc60", "a158_ma10_60", "a158_macd", "a191_pat_13",
    "a101_a005", "a191_rev_09", "a101_a004", "fund_gpm",
]
REBOUND_KEYS = [
    "dp_historical_n12_delta_window5", "dp_historical_n8_delta_window5", "dp_historical_n4_delta_window5",
    "SP_n10", "SP_n60", "SP_n5", "SP_n120", "SP_n20", "senti", "fund_turnover", "a191_liq_04",
    "std_turn_6m", "a191_liq_11", "a101_a040", "a101_a018", "a191_mom_18", "a191_mom_19",
    "a191_vol_01", "a101_a032", "a101_a054", "a191_vol_25", "volume_variance_n10_delta_window5",
    "a191_vol_10", "a191_hof_04", "a101_a063", "a101_a090", "a101_a057", "a101_a015", "a101_a075",
    "a101_a087", "a158_low20", "a158_resi20", "a158_roc20", "fund_earn_mom", "sales_growth_n8_ma_window20",
    "sales_growth_n4_ma_window20", "a191_rev_07", "north_buy_prop", "open_net_buy_intensity",
    "heavyweight_overweight_n10", "a158_ma5_20",
]


def run(name, keys):
    return rolling_train_dual(
        pool="csi300", pool_label=name, start_anchor="2024-04-01", end_anchor="2025-07-01",
        step_months=3, train_years=5, valid_months=6, factor_set="WhiteBoxHandler",
        custom_factor_keys=keys, label_horizon=20, label_norm="csrank",
        label_mode="tradable_open_to_open", early_stop_metric="rank_ic", param_profile="optuna",
        enable_winsorize=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = {"status": "completed", "contract": {"start_anchor": "2024-04-01", "end_anchor": "2025-07-01", "step_months": 3, "label_horizon": 20, "label_mode": "tradable_open_to_open", "pool": "csi300"}}
    result["trend"] = run("formal_trend", TREND_KEYS)
    result["rebound"] = run("formal_rebound", REBOUND_KEYS)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: {"n_planned": v.get("n_planned"), "n_trained": v.get("n_trained")} for k, v in result.items() if isinstance(v, dict) and "n_planned" in v}, ensure_ascii=False))
