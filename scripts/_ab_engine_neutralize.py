"""引擎级 A/B: 行业中性化开/关 (真实回测模拟器)。

用同一个 DirectionBacktestRunner 实例跑两次 (prediction 只加载一次, runner 内部缓存):
- run(neutralize_qlib=True,  macro_bias=False, sector_fusion=False, fixed_model="trend_momentum")
- run(neutralize_qlib=False, macro_bias=False, sector_fusion=False, fixed_model="trend_momentum")
输出 summary 到 stdout + 写 JSON 结果文件。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/experiments/ab_engine_neutralize.json"

METRICS = ["total_return", "annual_return", "annual_excess", "sharpe", "max_drawdown",
           "win_rate", "excess_win_rate", "calmar", "volatility", "turnover"]

def run():
    t0 = time.time()
    from core.domain.backtest.engine import DirectionBacktestRunner
    runner = DirectionBacktestRunner()
    results = {}
    for neu in (True, False):
        tag = "neu" if neu else "no_neu"
        print(f"[AB] start neutralize_qlib={neu} t+{time.time()-t0:.0f}s", flush=True)
        r = runner.run(
            start="2026-01-01", end="2026-08-14",
            pool="a500", top_n=10, freq="week",
            use_rule_based=False, use_crf=True, use_vol_switch=False,
            macro_bias=False, sector_fusion=False,
            stop_source="fixed",
            fixed_model=["656749665990832865", "697458c44bdf4568ab99bafcf9eb9636"],
            neutralize_qlib=neu,
        )
        s = r.get("summary", r) if isinstance(r, dict) else {}
        results[tag] = {k: s.get(k) for k in METRICS}
        results[tag]["score_processing"] = r.get("score_processing") if isinstance(r, dict) else None
        results[tag]["model_routing"] = r.get("model_routing") if isinstance(r, dict) else None
        results[tag]["n_periods"] = len(r.get("periods", [])) if isinstance(r, dict) else 0
        print(f"[AB] done neutralize_qlib={neu} total_return={results[tag].get('total_return')}", flush=True)
    results["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1), flush=True)
    print("[AB] DONE", flush=True)

if __name__ == "__main__":
    run()
