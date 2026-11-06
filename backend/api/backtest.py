"""策略回测 API"""
import logging
import threading
from datetime import date
from typing import Dict, List, Optional, Literal

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from core.shared.infrastructure.async_task import task_manager
from backend.service.qlib_access import run_with_qlib_read_access

logger = logging.getLogger(__name__)
router = APIRouter()

JOBS = {}


class DirectionBacktestRequest(BaseModel):
    start: str
    end: str
    pool: str = "stable_history_500d"
    profile_id: Optional[str] = None
    profile_version: Optional[int] = Field(None, ge=1)
    top_n: int = Field(5, ge=1, le=500)
    freq: Literal["day", "2d", "3d", "4d", "5d", "10d", "20d", "week", "biweek", "3w", "month"] = "week"
    sizing: Literal["equal", "score_weighted"] = "equal"
    target_exposure: float = Field(0.95, gt=0, le=1)
    max_one_way_turnover: float = Field(0.30, ge=0, le=1)
    directions: List[Literal["trend_momentum", "volume_breakout", "oversold_rebound", "value_defense"]] = ["trend_momentum"]
    use_hmm: bool = False
    use_timing: bool = False
    full_position: bool = True
    use_rule_based: bool = False
    use_crf: bool = False
    use_vol_switch: bool = False
    use_conditional_switch: bool = True
    conditional_fallback_policy: Literal["error", "binary"] = "error"
    vol_label_horizon: int = Field(10, ge=1, le=60)
    state_models: Dict[str, List[str]] = Field(default_factory=dict)
    fixed_model: Optional[List[str]] = None
    code_prefixes: Optional[List[str]] = Field(default_factory=lambda: [
        "600", "601", "603", "605", "000", "001", "002", "003",
    ])
    commission_rate: Optional[float] = Field(None, ge=0, le=0.05)
    slippage_rate: Optional[float] = Field(None, ge=0, le=0.05)
    stamp_duty: Optional[float] = Field(None, ge=0, le=0.05)
    stop_loss_pct: Optional[float] = Field(None, ge=-1, lt=0)
    take_profit_pct: Optional[float] = Field(None, gt=0, le=10)
    stop_loss_warn: Optional[float] = Field(None, ge=-1, lt=0)
    stop_loss_full: Optional[float] = Field(None, ge=-1, lt=0)
    take_profit_half: Optional[float] = Field(None, gt=0, le=10)
    take_profit_full: Optional[float] = Field(None, gt=0, le=10)
    max_position_pct: Optional[float] = Field(None, gt=0, le=1)
    max_sector_pct: Optional[float] = Field(None, gt=0, le=1)
    cash_yield: bool = False
    stop_source: Literal["off", "fixed", "regime", "dynamic", "atr"] = "off"
    dynamic_lookback: int = Field(20, ge=2, le=252)
    dynamic_trail: float = Field(0.05, gt=0, lt=1)
    atr_period: int = Field(14, ge=2, le=60)
    atr_mult: float = Field(2.0, gt=0, le=10)
    macro_bias: bool = False
    macro_regime: bool = False
    neutralize_qlib: bool = False
    sector_fusion: bool = False
    decision_day_filter: bool = True
    cooldown_stop_weeks: Optional[int] = Field(None, ge=0, le=52)
    cooldown_profit_weeks: Optional[int] = Field(None, ge=0, le=52)
    decision_policy: Optional[dict] = None
    decision_overlays: Optional[Dict[str, Dict[str, float]]] = None

    @model_validator(mode="after")
    def validate_dates_and_caps(self):
        if date.fromisoformat(self.start) > date.fromisoformat(self.end):
            raise ValueError("回测开始日期不能晚于结束日期")
        if self.max_position_pct is not None and self.max_position_pct > self.target_exposure:
            raise ValueError("单只仓位上限不能超过目标总仓位")
        if (self.max_position_pct is not None and self.max_sector_pct is not None
                and self.max_position_pct > self.max_sector_pct):
            raise ValueError("单只仓位上限不能超过行业仓位上限")
        if (self.stop_loss_warn is not None and self.stop_loss_full is not None
                and self.stop_loss_full >= self.stop_loss_warn):
            raise ValueError("全额止损阈值必须低于预警止损阈值")
        if (self.take_profit_half is not None and self.take_profit_full is not None
                and self.take_profit_full <= self.take_profit_half):
            raise ValueError("全额止盈阈值必须高于半额止盈阈值")
        return self


@router.post("/direction")
async def start_direction_backtest(req: DirectionBacktestRequest):
    """启动策略方向回测（异步）"""

    from backend.service.backtest import BacktestService
    try:
        BacktestService.validate_direction_request(req)
    except ValueError as e:
        raise HTTPException(422, str(e))

    def _run():
        heartbeat_stop = threading.Event()

        def _heartbeat():
            while not heartbeat_stop.wait(30):
                task_manager.heartbeat(job_id, {
                    "stage": "backtest",
                    "message": "策略回测计算中",
                })

        heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"backtest-heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            return BacktestService.run_direction(req)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)

    job_id = task_manager._make_id()
    task_manager.submit_with_id(
        job_id,
        lambda: run_with_qlib_read_access(_run),
        description=f"策略回测 {req.start}~{req.end} pool={req.pool}",
        pool="default",
    )
    JOBS[job_id] = {"req": req.dict(), "created": True}
    return {"data": {"job_id": job_id, "status": "started"}}


@router.get("/direction/{job_id}")
async def get_direction_backtest_result(job_id: str):
    """获取方向回测任务结果"""
    task = await anyio.to_thread.run_sync(task_manager.get_dict, job_id)
    if task is None:
        raise HTTPException(404, f"任务 {job_id} 不存在")

    status = task["status"]
    if status in ("pending", "running", "cancel_requested"):
        return {
            "status": "running",
            "progress": task.get("progress", {}),
            "message": task.get("description", ""),
        }
    elif status == "stalled":
        return {
            "status": "stalled",
            "progress": task.get("progress", {}),
            "message": task.get("progress", {}).get(
                "message", "超过 10 分钟未收到心跳，任务仍可能在后台运行"
            ),
            "error": task.get("error", ""),
        }
    elif status == "completed":
        result = task["result"]
        if isinstance(result, dict):
            result["status"] = "done"
        return result
    elif status == "failed":
        return {"status": "error", "error": task.get("error") or "回测任务失败"}
    elif status == "cancelled":
        return {"status": "error", "error": "回测任务已取消"}
    else:
        return {"status": "error", "error": task.get("error") or "回测任务已中断"}


@router.get("/training-cutoffs")
async def get_training_cutoffs():
    """返回各模型的训练截止日期，前端用于前视偏差警告

    从实际模型文件读取，不硬编码日期。
    - qlib_fit_end: Qlib 模型 z-score 归一化的 fit_end_time
    - qlib_train_end: Qlib 模型训练集的结束日期
    - timing_train_end: 择时模型的训练结束日期
    - safe_start: 建议的最早回测起始日期（三者取最大 + 1天）
    """
    def _run():
        import datetime as dt
        from pathlib import Path

        qlib_fit_end = None
        qlib_train_end = None
        timing_train_end = None

        # 从当前活跃方向模型 + training_results 表获取训练截止日期
        try:
            from core.data import SettingsRepo, Database
            direction_models = SettingsRepo.get("direction_models")
            if not direction_models:
                # 回退: 读磁盘缓存
                import json
                cache_file = Path(".cache/direction_models.json")
                if cache_file.exists():
                    direction_models = json.loads(cache_file.read_text())

            if direction_models:
                latest_fit = None
                latest_train = None
                for dir_name, (exp_id, run_id) in direction_models.items():
                    rows = Database.fetchall(
                        "SELECT train_start, train_end, valid_start, valid_end, "
                        "test_start, test_end FROM training_results "
                        "WHERE run_id = %s ORDER BY created_at DESC LIMIT 1",
                        (run_id,))
                    if rows and rows[0].get("train_end"):
                        te = rows[0]["train_end"]
                        d = te if isinstance(te, dt.date) else dt.date.fromisoformat(str(te)[:10])
                        if latest_train is None or d > latest_train:
                            latest_train = d
                        if latest_fit is None or d > latest_fit:
                            latest_fit = d
                if latest_fit:
                    qlib_fit_end = latest_fit.isoformat()
                if latest_train:
                    qlib_train_end = latest_train.isoformat()
            else:
                logger.info("[cutoffs] 未找到活跃方向模型, 跳过 Qlib 日期扫描")
        except Exception as e:
            logger.warning(f"读取 Qlib 训练日期失败: {e}")

        try:
            from core.domain.regime.predictor import get_regime_predictor
            rp = get_regime_predictor()
            if rp.last_train_date:
                timing_train_end = str(rp.last_train_date)[:10]
        except Exception as e:
            logger.warning(f"读取 RegimePredictor 训练日期失败: {e}")

        safe_start = None
        candidates = []
        if qlib_fit_end:
            candidates.append(dt.date.fromisoformat(qlib_fit_end))
        if timing_train_end:
            candidates.append(dt.date.fromisoformat(timing_train_end))
        if candidates:
            safe_start = (max(candidates) + dt.timedelta(days=1)).isoformat()

        return {
            "qlib_fit_end": qlib_fit_end,
            "qlib_train_end": qlib_train_end,
            "timing_train_end": timing_train_end,
            "safe_start": safe_start,
        }
    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}
