import logging
import os
import anyio
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from core.shared.infrastructure.async_task import task_manager
from core.domain.timing.breadth import get_breadth as _get_breadth, get_breadth_history as _get_breadth_history

logger = logging.getLogger(__name__)
router = APIRouter()

_training_lock = __import__('threading').Lock()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def _verify_admin(authorization: str | None) -> bool:
    if not ADMIN_TOKEN:
        logger.warning("ADMIN_TOKEN 未配置, 拒绝所有管理操作")
        return False
    if not authorization:
        return False
    token = authorization.removeprefix("Bearer ").strip()
    if token == ADMIN_TOKEN:
        return True
    # 安全: 仅认 ADMIN_TOKEN, 不把任意登录 JWT 当管理员
    return False

class TrainRequest(BaseModel):
    label_horizon: int = 10
    threshold: float = 0.01
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    valid_start: Optional[str] = None
    valid_end: Optional[str] = None
    model_type: str = "tree"
    task_type: str = "binary"  # "binary" 择时 | "regime" V1分类 | "regime_v2" 回归版


def _resolve_train_dates(req):
    """训练区间动态默认: 缺省时用最新交易日往前推 (train 6年 + valid 最近180天)"""
    from datetime import timedelta
    from core.shared.persistence import today_date
    end = today_date()
    valid_end = req.valid_end or end.isoformat()
    valid_start = req.valid_start or (end - timedelta(days=180)).isoformat()
    train_end = req.train_end or (end - timedelta(days=181)).isoformat()
    train_start = req.train_start or (end - timedelta(days=181 + 365 * 6)).isoformat()
    return train_start, train_end, valid_start, valid_end


class PredictRequest(BaseModel):
    as_of_date: Optional[str] = None


class BacktestRequest(BaseModel):
    start_date: str = "2020-01-01"
    end_date: str = "2026-07-01"
    benchmark: str = "sh000300"
    model_name: Optional[str] = None


def _present_breadth_metrics(result):
    if not isinstance(result, dict):
        return result
    data = dict(result)
    for public_key in (
        "up_count", "down_count", "flat_count", "unknown_count",
        "valid_change_count", "change_coverage", "up_ratio", "up_volume_ratio",
        "pct_median", "amplitude_median",
    ):
        observed_key = f"observed_{public_key}"
        if observed_key in data:
            data[public_key] = data[observed_key]
    for key in ("up_volume_ratio", "amplitude_median"):
        observed_key = f"observed_{key}"
        if observed_key in data:
            data[key] = data[observed_key]
    if "observed_row_count" in data:
        data["total_stocks"] = data["observed_row_count"]
    data["limit_up"] = None
    data["limit_down"] = None
    data["limit_count_status"] = "unavailable"
    data["limit_count_missing_reason"] = "security_specific_limit_price_unavailable"
    data["coverage_basis"] = "observed_daily_quotes_rows"
    return data


@router.get("/breadth")
async def get_breadth_api():
    """获取最新市场宽度指标"""
    result = await anyio.to_thread.run_sync(_get_breadth)
    return {"data": _present_breadth_metrics(result)}


@router.get("/breadth/history")
async def get_breadth_history_api(start: str, end: str):
    """获取历史市场宽度序列"""
    df = await anyio.to_thread.run_sync(_get_breadth_history, start, end)
    if df.empty:
        raise HTTPException(404, "暂无数据")
    records = [_present_breadth_metrics(row) for row in df.reset_index().to_dict("records")]
    return {"data": records}


@router.post("/train")
async def train_timing(req: TrainRequest, authorization: str | None = Header(None)):
    """异步训练择时模型 (同一时间只允许一个训练任务)"""
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权"]}}, status_code=401)
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(409, "训练任务正在进行中, 请稍后再试")

    from core.domain.timing.model import IndexTimingModel
    from core.domain.timing import model as _timing

    def _do_train():
        try:
            train_start, train_end, valid_start, valid_end = _resolve_train_dates(req)
            if req.task_type == "regime":
                from core.domain.regime.predictor import RegimePredictor, get_regime_predictor
                model = RegimePredictor(label_horizon=req.label_horizon, model_type=req.model_type)
                result = model.fit(
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                from core.domain.regime import predictor as _rp
                _rp._regime_predictor = model
                return result
            elif req.task_type == "regime_v2":
                from core.domain.regime.predictor_v2 import RegimePredictorV2, get_regime_predictor_v2
                model = RegimePredictorV2(label_horizon=req.label_horizon)
                result = model.fit(
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                from core.domain.regime import predictor_v2 as _rpv2
                _rpv2._regime_predictor_v2 = model
                return result
            else:
                model = IndexTimingModel(label_horizon=req.label_horizon, threshold=req.threshold, model_type=req.model_type)
                result = model.fit(
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                _timing._cached_model = None
                return result
        finally:
            _training_lock.release()

    try:
        job_id = task_manager.submit(_do_train, description=f"择时模型训练 h={req.label_horizon}", pool="default")
    except Exception:
        _training_lock.release()
        raise
    return {"data": {"job_id": job_id, "status": "started"}}


@router.get("/train/{job_id}")
async def train_status(job_id: str):
    """查询训练任务状态"""
    task = task_manager.get_dict(job_id)
    if task is None:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    return {"data": task}


@router.post("/predict")
async def predict_timing(req: PredictRequest):
    """获取当前择时信号"""
    from core.domain.timing.model import get_model
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(503, f"模型加载失败: {e}")
    if model.lgb is None:
        raise HTTPException(503, "模型未训练或预测失败，请先训练模型")
    result = await anyio.to_thread.run_sync(lambda: model.predict(as_of=req.as_of_date))
    if result.get("prob") is None:
        raise HTTPException(503, result.get("error") or "模型预测失败")
    return {"data": result}


@router.post("/regime/predict")
async def predict_regime(req: PredictRequest, authorization: str | None = Header(None)):
    """获取市场状态 — V1 方向 + V2 风控集成"""
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权"]}}, status_code=401)
    from core.domain.regime.classifier import get_regime_probs
    result = await anyio.to_thread.run_sync(get_regime_probs)
    if result is None or result.get("probs") is None:
        raise HTTPException(503, "状态预测失败，请先训练 regime 模型")
    return {"data": result}


@router.get("/regime/current")
async def current_regime(apply_macro: bool = Query(True), smooth: bool = Query(True)):
    """获取当前市场状态 (包含完整概率分布)"""
    from core.domain.regime.classifier import get_regime_probs, get_current_regime
    probs = get_regime_probs(apply_macro=apply_macro, smooth=smooth)
    if probs is not None:
        result = get_current_regime(apply_macro=apply_macro, smooth=smooth)
        return {"data": {"regime": result.to_dict(), "probs": probs["probs"],
                         "position_ratio": probs.get("position_ratio"),
                         "signal_weights": probs.get("signal_weights")}}
    result = get_current_regime(apply_macro=apply_macro, smooth=smooth)
    return {"data": {"regime": result.to_dict(), "probs": None, "position_ratio": None, "signal_weights": None}}


@router.get("/regime/model-info")
async def regime_model_info():
    """当前 regime 模型信息: 最后训练日期 + 学得转移矩阵 (供前端展示陈旧提醒/热力图)"""
    import numpy as np
    from core.domain.regime.predictor import get_regime_predictor
    rp = get_regime_predictor()
    if rp is None or not rp._is_loaded():
        return {"data": {"loaded": False}}
    tm = getattr(rp, "transition_matrix", None)
    return {"data": {
        "loaded": True,
        "model_type": rp.model_type,
        "label_horizon": rp.label_horizon,
        "last_train_date": str(rp.last_train_date) if rp.last_train_date else None,
        "valid_acc": round(rp.valid_acc, 4) if rp.valid_acc is not None else None,
        "valid_top2": round(rp.valid_top2, 4) if rp.valid_top2 is not None else None,
        "transition_matrix": (np.asarray(tm).tolist() if tm is not None else None),
    }}


# ── 仓位映射配置 ──

class RatioConfigRequest(BaseModel):
    ratios: dict
    stops: dict = {}

@router.get("/regime/ratio-config")
async def get_ratio_config():
    """获取当前仓位映射 + 止盈止损"""
    from core.domain.regime.predictor import RATIO_MAP, STATE_NAMES
    import json
    stops = {}
    try:
        cfg = json.loads((__import__('pathlib').Path('config/regime_ratio.json')).read_text())
        stops = cfg.get('stops', {})
    except: pass
    return {"data": {
        "ratios": {str(i): RATIO_MAP[i] for i in range(6)},
        "stops": {str(i): stops.get(str(i), {}) for i in range(6)},
        "labels": {str(i): STATE_NAMES[i] for i in range(6)},
    }}

@router.post("/regime/ratio-config")
async def set_ratio_config(req: RatioConfigRequest, authorization: str | None = Header(None)):
    """更新仓位映射 (立即生效)"""
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权"]}}, status_code=401)
    from core.domain.regime.predictor import save_ratio_config, STATE_NAMES
    from fastapi import HTTPException
    ratios = {}
    for k, v in req.ratios.items():
        idx = int(k)
        val = float(v)
        if idx < 0 or idx >= len(STATE_NAMES):
            raise HTTPException(400, f"非法市场状态索引: {k} (允许 0~{len(STATE_NAMES)-1})")
        if val < 0 or val > 100:
            raise HTTPException(400, f"仓位比例需在 0~100 之间: {k}={v}")
        ratios[idx] = val
    stops = {}
    if req.stops:
        stops = {str(k): v for k, v in req.stops.items()}
    save_ratio_config(ratios, stops)
    return {"data": {"status": "ok"}}


@router.post("/backtest")
async def backtest_timing(req: BacktestRequest, authorization: str | None = Header(None)):
    """异步回测择时策略"""
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权"]}}, status_code=401)
    from core.domain.timing.model import IndexTimingModel

    def _do_backtest():
        model = IndexTimingModel()
        if req.model_name:
            model._load_path(req.model_name)
        return model.backtest(start=req.start_date, end=req.end_date)

    job_id = task_manager.submit(_do_backtest, description=f"择时回测 {req.start_date}~{req.end_date}", pool="default")
    return {"data": {"job_id": job_id, "status": "started"}}


@router.delete("/model/{model_name}")
async def delete_timing_model(model_name: str, authorization: str | None = Header(None)):
    """删除择时/Regime模型文件"""
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权"]}}, status_code=401)
    from pathlib import Path
    base = Path("data/timing_model")
    # 安全检查
    if ".." in model_name or "/" in model_name or "\\" in model_name:
        raise HTTPException(400, "无效的模型名")
    # 允许不带 .pkl 后缀
    path = base / model_name
    if not path.exists():
        path = base / (model_name + ".pkl")
    if not path.exists():
        raise HTTPException(404, f"模型 {model_name} 不存在")
    try:
        path.unlink()
        return {"data": {"deleted": path.name}}
    except Exception as e:
        raise HTTPException(500, f"删除失败: {e}")


@router.get("/backtest/{job_id}")
async def backtest_status(job_id: str):
    """查询回测任务状态"""
    task = task_manager.get_dict(job_id)
    if task is None:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    return {"data": task}
