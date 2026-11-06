import numpy as np
import pandas as pd


def percentile_rank(scores):
    ranked = scores.groupby(level="datetime").rank(pct=True)
    if ranked.isna().any() or not np.isfinite(ranked.to_numpy(dtype=float)).all():
        raise ValueError("预测分数必须完整且有限")
    return ranked


def fit_rank_weight(lgb_rank, lstm_rank, label_percentile, center=0.5, penalty=0.01):
    if not lgb_rank.index.equals(lstm_rank.index) or not lgb_rank.index.equals(label_percentile.index):
        raise ValueError("校准样本索引不一致")
    frame = pd.DataFrame({"g": lgb_rank, "n": lstm_rank, "y": label_percentile})
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("校准分数或标签非有限")
    delta = frame["n"] - frame["g"]
    residual = frame["y"] - frame["g"]
    day_count = frame.groupby(level="datetime")["g"].transform("size")
    weights = 1.0 / day_count.to_numpy(dtype=float)
    total = frame.index.get_level_values("datetime").nunique()
    numerator = np.dot(weights, (delta * residual).to_numpy(dtype=float)) / total + penalty * center
    denominator = np.dot(weights, (delta * delta).to_numpy(dtype=float)) / total + penalty
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def blend_rank(lgb_rank, lstm_rank, weight):
    if not lgb_rank.index.equals(lstm_rank.index) or not 0 <= weight <= 1:
        raise ValueError("融合索引或权重无效")
    return (1.0 - weight) * lgb_rank + weight * lstm_rank


def fit_residual_weight(base_rank, residual_prediction, label_percentile, penalty=0.01):
    if not base_rank.index.equals(residual_prediction.index) or not base_rank.index.equals(label_percentile.index):
        raise ValueError("残差校准样本索引不一致")
    frame = pd.DataFrame({"base": base_rank, "residual": residual_prediction, "label": label_percentile})
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("残差校准样本非有限")
    day_count = frame.groupby(level="datetime")["base"].transform("size")
    weights = 1.0 / day_count.to_numpy(dtype=float)
    total = frame.index.get_level_values("datetime").nunique()
    residual = frame["residual"].to_numpy(dtype=float)
    target = (frame["label"] - frame["base"]).to_numpy(dtype=float)
    numerator = np.dot(weights, residual * target) / total
    denominator = np.dot(weights, residual * residual) / total + penalty
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def mature_update_dates(calendar, label_end, decision_date, lookback=126):
    cutoff = pd.Timestamp(decision_date)
    eligible = calendar[calendar < cutoff]
    eligible = eligible[-lookback:]
    mature = label_end.reindex(eligible)
    return eligible[mature.notna().to_numpy() & (mature.to_numpy() < cutoff.to_datetime64())]
