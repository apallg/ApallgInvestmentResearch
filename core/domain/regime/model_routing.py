VOLATILITY_MODEL_STATES = ("high_vol", "low_vol")
REGIME_MODEL_STATES = (
    "volatile_up", "volatile_down", "volatile_flat",
    "steady_up", "steady_down", "steady_flat",
)


def normalize_state_models(state_models):
    import re

    normalized = {}
    for key, value in (state_models or {}).items():
        if isinstance(value, str) and "/" in value:
            value = value.split("/", 1)
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(value):
            raise ValueError(f"无效模型映射: {key}")
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", str(item)) for item in value):
            raise ValueError(f"无效模型标识: {key}")
        normalized[str(key)] = (str(value[0]), str(value[1]))
    return normalized


def normalize_model_ref(model):
    if not model:
        return None
    return normalize_state_models({"fixed": model})["fixed"]


def validate_state_models(state_models, use_vol_switch):
    normalized = normalize_state_models(state_models)
    if not normalized:
        return {}
    expected = set(VOLATILITY_MODEL_STATES if use_vol_switch else REGIME_MODEL_STATES)
    missing = expected - set(normalized)
    extra = set(normalized) - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"缺少: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"无效: {', '.join(sorted(extra))}")
        raise ValueError("状态模型映射不完整（" + "；".join(parts) + "）")
    return normalized


def resolve_state_key(regime, direction, use_vol_switch):
    if use_vol_switch:
        if direction not in ("oversold_rebound", "trend_momentum"):
            raise ValueError(f"无效波动率方向: {direction}")
        return "high_vol" if direction == "oversold_rebound" else "low_vol"
    return str(regime)
