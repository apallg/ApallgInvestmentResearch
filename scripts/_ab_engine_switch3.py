"""引擎 A/B: 二值波动率切换 vs 三档条件切换 (只选模型, 不动仓位)。

同一 runner 实例跑两次 (predictions 只加载一次, runner 内部缓存):
  1. binary : use_vol_switch=True, use_conditional_switch=False (现状)
  2. cond   : use_vol_switch=True, use_conditional_switch=True (三档模型选择)
窗口/输出由命令行参数指定。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/experiments/ab_engine_switch3.json"

TREND = ("656749665990832865", "697458c44bdf4568ab99bafcf9eb9636")
REBOUND = ("656749665990832865", "32f8ca7de0a04196ac1095ad31afcfdc")
STATE_MODELS = {"low_vol": TREND, "high_vol": REBOUND}

METRICS = ["total_return", "annual_return", "annual_excess", "sharpe", "max_drawdown",
           "win_rate", "excess_win_rate", "calmar"]


def run(start, end):
    t0 = time.time()
    from core.domain.backtest.engine import DirectionBacktestRunner
    runner = DirectionBacktestRunner()
    results = {}
    cases = [
        ("binary", dict(use_vol_switch=True, use_conditional_switch=False)),
        ("cond", dict(use_vol_switch=True, use_conditional_switch=True)),
    ]
    for tag, kw in cases:
        print(f"[AB] start {tag} t+{time.time()-t0:.0f}s", flush=True)
        r = runner.run(
            start=start, end=end,
            pool="a500", top_n=10, freq="week",
            use_rule_based=False, use_crf=True,
            macro_bias=False, sector_fusion=False,
            stop_source="fixed", state_models=STATE_MODELS,
            **kw,
        )
        s = r.get("summary", r) if isinstance(r, dict) else {}
        results[tag] = {k: s.get(k) for k in METRICS}
        results[tag]["n_periods"] = len(r.get("periods", [])) if isinstance(r, dict) else 0
        if isinstance(r, dict):
            results[tag]["model_usage"] = r.get("model_routing", {}).get("usage")
            results[tag]["score_processing"] = r.get("score_processing")
        # 记录每期 direction/regime 分布
        if isinstance(r, dict) and r.get("periods"):
            from collections import Counter
            results[tag]["direction_counts"] = dict(Counter(p.get("direction") for p in r["periods"]))
            results[tag]["regime_counts"] = dict(Counter(p.get("regime") for p in r["periods"]))
        print(f"[AB] done {tag} total_return={results[tag].get('total_return')} "
              f"usage={results[tag].get('model_usage')} "
              f"cond_env={len(getattr(runner, '_cond_env', {}))}", flush=True)
    results["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1), flush=True)
    print("[AB] DONE", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    globals()["OUT"] = Path(args.out)
    run(args.start, args.end)
