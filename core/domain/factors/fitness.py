import math


def summarize_ic_series(values, total_count=None):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "mean": None, "std": None, "icir": None, "positive_ratio": 0.0,
            "coverage": 0.0, "count": 0,
        }
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    std = math.sqrt(variance)
    denominator = int(total_count) if total_count is not None else len(values)
    return {
        "mean": mean,
        "std": std,
        "icir": mean / std if std > 1e-12 else (mean if mean else 0.0),
        "positive_ratio": sum(v > 0 for v in clean) / len(clean),
        "coverage": len(clean) / denominator if denominator > 0 else 0.0,
        "count": len(clean),
    }


def compute_training_fitness(metrics, complexity=0, max_complexity=40):
    ic = abs(float(metrics.get("mean") or metrics.get("ic") or 0.0))
    icir = abs(float(metrics.get("icir") or 0.0))
    positive = float(metrics.get("positive_ratio") or 0.0)
    coverage = float(metrics.get("coverage") or 0.0)
    penalty = min(float(complexity) / max_complexity, 1.0) if max_complexity else 0.0
    return 0.55 * ic + 0.20 * min(icir, 2.0) / 2.0 + 0.15 * positive + 0.10 * coverage - 0.08 * penalty


def evaluate_factor_portfolio(scores_by_date, returns_by_date, quantile=0.2, cost_bps=10):
    daily = []
    for date in sorted(set(scores_by_date) & set(returns_by_date)):
        scores = scores_by_date.get(date) or {}
        returns = returns_by_date.get(date) or {}
        rows = [(code, float(score), float(returns[code])) for code, score in scores.items()
                if code in returns and score is not None and returns[code] is not None
                and math.isfinite(float(score)) and math.isfinite(float(returns[code]))]
        if len(rows) < 10:
            continue
        rows.sort(key=lambda row: row[1])
        n = max(1, int(len(rows) * quantile))
        long_ret = sum(row[2] for row in rows[-n:]) / n
        short_ret = sum(row[2] for row in rows[:n]) / n
        daily.append(long_ret - short_ret - cost_bps / 10000.0)
    if not daily:
        return {"net_return": None, "sharpe": None, "max_drawdown": None, "turnover": None, "days": 0}
    equity = 1.0
    curve = [equity]
    for value in daily:
        equity *= 1.0 + value
        curve.append(equity)
    mean = sum(daily) / len(daily)
    variance = sum((v - mean) ** 2 for v in daily) / len(daily)
    std = math.sqrt(variance)
    peak = curve[0]
    drawdowns = []
    for value in curve:
        peak = max(peak, value)
        drawdowns.append(value / peak - 1.0)
    return {
        "net_return": equity - 1.0,
        "sharpe": mean / std * math.sqrt(252) if std > 1e-12 else 0.0,
        "max_drawdown": min(drawdowns),
        "turnover": 2.0,
        "days": len(daily),
    }
