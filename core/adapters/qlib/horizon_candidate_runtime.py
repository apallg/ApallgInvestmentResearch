import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn


class CandidateLSTM(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.daily = nn.LSTM(features + 1, 32, num_layers=1, batch_first=True)
        self.output = nn.Linear(32, 1)

    def forward(self, daily):
        return self.output(self.daily(daily)[0][:, -1, :]).squeeze(-1)


def load_candidate(path, horizon):
    package = json.loads(Path(path).read_text(encoding="utf-8"))
    if package.get("status") != "research_candidate_packaged":
        raise RuntimeError("候选包尚未完成研究封装")
    profiles = [row for row in package["decisions"] if row["horizon"] == horizon]
    if len(profiles) != 1:
        raise RuntimeError("期限候选配置缺失或重复")
    profile = profiles[0]
    members = []
    for member in profile["members"]:
        components = {}
        for name, record in member["artifacts"].items():
            artifact = Path(record["path"])
            if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"候选组件签名不符: {artifact}")
            if artifact.suffix == ".txt":
                model = lgb.Booster(model_str=artifact.read_text(encoding="utf-8"))
                if model.num_feature() != len(profile["factor_keys"]):
                    raise RuntimeError(f"候选LGB特征维度不符: {artifact}")
                components[name] = ("lgb", model)
            else:
                state = torch.load(artifact, map_location="cpu", weights_only=False)
                if state.get("daily_features") != len(profile["factor_keys"]) + 1:
                    raise RuntimeError(f"候选LSTM特征维度不符: {artifact}")
                model = CandidateLSTM(len(profile["factor_keys"]))
                model.load_state_dict(state["model"])
                model.eval()
                components[name] = ("lstm", model, state["mean"], state["scale"])
        members.append({"seed": member["seed"], "components": components, "weight": member["weight"]})
    return {"profile": profile, "members": members}


def _sequence_scores(model, mean, scale, frame, expected_index, calendar):
    sessions = pd.DatetimeIndex(calendar)
    values = frame.to_numpy(dtype=np.float32)
    if len(mean) != values.shape[1] or len(scale) != values.shape[1]:
        raise RuntimeError("候选LSTM标准化参数不符")
    scores = []
    for start in range(0, len(expected_index), 4096):
        batch_index = expected_index[start:start + 4096]
        ends = sessions.get_indexer(batch_index.get_level_values("datetime"))
        if (ends < 0).any():
            raise RuntimeError("候选决策日不在交易日历")
        session_positions = ends[:, None] - np.arange(19, -1, -1)[None, :]
        valid_dates = (session_positions >= 0) & (session_positions < len(sessions))
        dates = sessions.take(np.clip(session_positions, 0, len(sessions) - 1).ravel())
        stocks = np.repeat(batch_index.get_level_values("instrument").to_numpy(), 20)
        lookup = pd.MultiIndex.from_arrays([dates, stocks], names=frame.index.names)
        positions = frame.index.get_indexer(lookup).reshape(-1, 20)
        positions[~valid_dates] = -1
        observed = positions >= 0
        daily = np.zeros((len(batch_index), 20, values.shape[1] + 1), dtype=np.float32)
        daily[:, :, :-1] = (values[np.maximum(positions, 0)] - mean) / scale
        daily[:, :, :-1] *= observed[:, :, None]
        daily[:, :, -1] = observed.astype(np.float32)
        with torch.no_grad():
            scores.append(model(torch.from_numpy(daily)).numpy())
    return pd.Series(np.concatenate(scores).astype(np.float32), index=expected_index)


def predict_candidate(candidate, frame, expected_index, calendar):
    profile = candidate["profile"]
    features = profile["factor_keys"]
    if list(frame.columns) != features or frame.index.has_duplicates or not expected_index.isin(frame.index).all():
        raise RuntimeError("候选输入契约或股票日分母不符")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise RuntimeError("候选输入存在缺失或非有限值")
    scores = []
    for member in candidate["members"]:
        components = {}
        for name, model_data in member["components"].items():
            if model_data[0] == "lgb":
                raw = model_data[1].predict(frame.loc[expected_index].to_numpy(dtype=np.float32), num_threads=4)
                components[name] = pd.Series(raw, index=expected_index)
            else:
                components[name] = _sequence_scores(model_data[1], model_data[2], model_data[3], frame, expected_index, calendar)
        ranks = {name: raw.groupby(level="datetime").rank(pct=True) for name, raw in components.items()}
        structure = profile["structure"]
        if structure in ("G", "N", "AX"):
            score = ranks[structure]
        elif structure in ("F50", "FC", "FR"):
            weight = 0.5 if structure == "F50" else member["weight"]
            if structure == "FR":
                by_month = {row["month"]: row["weight"] for row in weight}
                months = expected_index.get_level_values("datetime").to_period("M").astype(str)
                if any(month not in by_month for month in months):
                    raise RuntimeError("动态权重状态尚未覆盖预测月")
                weight = pd.Series([by_month[month] for month in months], index=expected_index)
            score = (1 - weight) * ranks["G"] + weight * ranks["N"]
        elif structure == "R":
            score = ranks["AX"] + member["weight"] * components["R"]
        else:
            raise RuntimeError(f"未知候选结构: {structure}")
        if not np.isfinite(score.to_numpy(dtype=float)).all():
            raise RuntimeError("候选预测非有限")
        scores.append(score)
    result = sum(scores) / len(scores)
    if not result.index.equals(expected_index):
        raise RuntimeError("候选预测分母变化")
    return result
