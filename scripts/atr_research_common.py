import ast
import datetime as dt
from pathlib import Path


DIRECTIONS = ["trend_momentum", "volume_breakout", "oversold_rebound", "value_defense"]
METRICS = ["annual_return", "annual_excess", "sharpe", "max_drawdown",
           "win_rate", "excess_win_rate", "calmar", "total_return"]


def find_seen_data_overlaps(runner, start, end):
    from core.shared import config
    from core.shared.persistence import load_text

    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    base = Path(config.QLIB_MLFLOW_URI.replace("file:", ""))
    overlaps = []
    seen = set()
    for direction, model in runner._load_models_cached(DIRECTIONS).items():
        exp_id, run_id = model
        model_key = (exp_id, run_id)
        if model_key in seen:
            continue
        seen.add(model_key)
        params = base / exp_id / run_id / "params"
        parsed_splits = 0
        for split in ("train", "valid"):
            path = params / f"dataset.kwargs.segments.{split}"
            if not path.exists():
                continue
            try:
                seg_start, seg_end = ast.literal_eval(load_text(str(path)))
                seg_start_d = dt.date.fromisoformat(str(seg_start)[:10])
                seg_end_d = dt.date.fromisoformat(str(seg_end)[:10])
            except Exception:
                continue
            parsed_splits += 1
            if max(start_d, seg_start_d) <= min(end_d, seg_end_d):
                overlaps.append({
                    "direction": direction,
                    "model": f"{exp_id}/{run_id[:8]}",
                    "split": split,
                    "start": str(seg_start),
                    "end": str(seg_end),
                })
        if parsed_splits < 2:
            overlaps.append({
                "direction": direction,
                "model": f"{exp_id}/{run_id[:8]}",
                "split": "unknown",
                "start": "元数据缺失",
                "end": "无法验证样本外",
            })
    return overlaps


def require_oos_or_override(runner, start, end, allow_seen_data):
    overlaps = find_seen_data_overlaps(runner, start, end)
    if not overlaps:
        return
    print("检测到模型已见数据重叠或元数据不可验证:")
    for item in overlaps:
        print(f"  {item['direction']}: {item['split']} "
              f"{item['start']}~{item['end']} ({item['model']})")
    if not allow_seen_data:
        raise SystemExit("拒绝非样本外验证；仅做敏感性分析时显式传 --allow-seen-data")
    print("已显式允许重叠，本次结果只能视为敏感性分析。\n")


def print_diagnostics(result):
    diag = result.get("trade_diagnostics", {})
    print("  diagnostics        "
          f"stops={diag.get('stop_loss_count', 0)} "
          f"full={diag.get('stop_loss_full_count', 0)} "
          f"half={diag.get('stop_loss_half_count', 0)} "
          f"atr={diag.get('atr_stop_count', 0)} "
          f"take_profit={diag.get('take_profit_count', 0)} "
          f"gaps={diag.get('gap_through_count', 0)} "
          f"avg_stop={diag.get('avg_stop_return')} "
          f"avg_gap={diag.get('avg_gap_slippage')} "
          f"notional={diag.get('traded_notional', 0)} "
          f"cost={diag.get('transaction_cost', 0)}")
