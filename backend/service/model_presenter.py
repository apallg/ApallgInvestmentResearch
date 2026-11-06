"""模型展示名解析服务 — 从 common.py 迁出的纯函数。"""
import json
import logging
import re

logger = logging.getLogger(__name__)


def resolve_factor_label(m, run_dir):
    """解析 WhiteBoxHandler 模型的实际因子集名称"""
    fs = m.get("factor_set", "")
    if "WhiteBox" not in fs:
        if "Alpha158" in fs:
            return "Alpha158"
        if "Alpha360" in fs:
            return "Alpha360"
        return None

    factor_keys = None

    # 1. 双模型/融合 factor_keys.json
    #    qlib save_objects 会把对象 pickle 落盘: 兼容 纯JSON / pickle(dict) / pickle(json字符串) 三种形态
    fk_json = run_dir / "artifacts" / "factor_keys.json"
    if fk_json.exists():
        try:
            raw = fk_json.read_bytes()
            obj = None
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                import pickle
                obj = pickle.loads(raw)
            if isinstance(obj, str):
                obj = json.loads(obj)
            if isinstance(obj, dict):
                factor_keys = obj.get("custom_factor_keys", [])
        except Exception:
            pass

    # 2. 单模型 params 文件
    if not factor_keys:
        fk_param = run_dir / "params" / "dataset.kwargs.handler.kwargs.factor_keys"
        if fk_param.exists():
            try:
                val = fk_param.read_text().strip()
                factor_keys = [k.strip().strip("'\"") for k in val.strip("[]").split(",") if k.strip()]
            except Exception:
                pass

    if not factor_keys or len(factor_keys) == 0:
        return "白盒因子"

    # 匹配已知精选集
    try:
        from core.domain.factors.registry import get_pool
        pool = get_pool()
        curated = pool.state.get("curated", {})
        if curated:
            fk_set = set(factor_keys)
            best_name, best_overlap = None, 0
            for name, entry in curated.items():
                overlap = len(fk_set & set(entry.get("flat_list", [])))
                if overlap > best_overlap:
                    best_overlap, best_name = overlap, name
            if best_name and best_overlap >= len(fk_set) * 0.8:
                return f"精选:{best_name}"
    except Exception:
        pass

    return f"自选{len(factor_keys)}因子"


def make_display_name(m):
    """为模型生成唯一辨识名称，如: LGB_Alpha158_HS300_19-24"""
    mc = m.get("model_class", "")
    if m.get("is_dual"):
        model_short = "LGB+XGB"
    elif "LSTM" in mc and "LGB" in mc:
        model_short = "LGB+LSTM"
    elif "LGB" in mc:
        model_short = "LGB"
    elif "XGB" in mc:
        model_short = "XGB"
    elif "CatBoost" in mc:
        model_short = "CAT"
    else:
        model_short = mc.split(".")[-1] if mc else "?"

    fl = m.get("factor_label", "")
    factor_short = None
    if not fl or fl == "WhiteBoxHandler":
        pass
    elif fl.startswith("精选:"):
        factor_short = "精选"
    elif fl.startswith("自选"):
        factor_short = "自选"
    elif fl in ("Alpha158", "Alpha360"):
        factor_short = "α158" if fl == "Alpha158" else "α360"
    elif len(fl) > 8:
        factor_short = fl[:8]
    else:
        factor_short = fl

    if not factor_short:
        fs = m.get("factor_set", "")
        if "Alpha158" in fs:
            factor_short = "α158"
        elif "Alpha360" in fs:
            factor_short = "α360"
        elif "WhiteBox" in fs:
            factor_short = "WB"
        else:
            factor_short = fs.split(".")[-1] if fs else "?"

    ins = (m.get("instruments") or "").lower()
    pool_map = {"csi300": "HS300", "hs300": "HS300", "csi500": "ZZ500",
                "a500": "稳定历史池500D", "stable_history_500d": "稳定历史池500D", "all": "ALL", "small_cap": "小市值"}
    pool_short = ""
    for k, v in pool_map.items():
        if k in ins:
            pool_short = v
            break
    if not pool_short:
        m2 = re.match(r'^list_(\d+)_stocks$', ins, re.I)
        if m2:
            pool_short = f"A{m2.group(1)}"
        else:
            pool_short = ins[:12] if ins else "?"

    tp = m.get("train_period", "")
    years_match = re.findall(r'(\d{4})', tp)
    if len(years_match) >= 2:
        year_range = f"{years_match[0][2:]}-{years_match[-1][2:]}"
    elif len(years_match) == 1:
        year_range = years_match[0][2:]
    else:
        year_range = ""

    label = m.get("label_period")
    label_suffix = f"_{label}d" if label else ""

    ts = m.get("trained_at", "")
    time_suffix = ""
    if ts:
        try:
            t = ts[:16].replace("T", " ")
            time_suffix = f"@{t[5:7]}{t[8:10]}_{t[11:13]}{t[14:16]}"
        except Exception:
            pass

    expert_short = ""
    if m.get("training_mode") == "state_subset":
        expert_short = {"trend": "趋势子样本", "rebound": "反弹子样本"}.get(
            m.get("expert_role"), "状态子样本")
    parts = [model_short]
    if expert_short:
        parts.append(expert_short)
    parts.extend([factor_short, pool_short])
    if year_range:
        parts.append(year_range)
    name = "_".join(parts)
    if label_suffix and len(name) < 30:
        name += label_suffix
    if time_suffix:
        name += time_suffix
    return name
