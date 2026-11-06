"""Qlib 集成 API — 模型训练端点（薄接口，委托 TrainingService）"""
import anyio
from fastapi import APIRouter, BackgroundTasks

from backend.api.qlib.models import TrainRequest, OptunaSearchRequest, RollingTrainRequest, ModelToggleRequest, EnsembleTrainRequest
from backend.service.training import TrainingService

router = APIRouter()


@router.get("/models")
async def list_models():
    """列出可用的模型类型"""
    return TrainingService.list_models()


@router.post("/models/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """启动模型训练（异步）"""
    return await anyio.to_thread.run_sync(TrainingService.submit_train, req, background_tasks)


@router.post("/models/ensemble-train")
async def ensemble_train(req: EnsembleTrainRequest, background_tasks: BackgroundTasks):
    """多模型集成训练（异步）"""
    return await anyio.to_thread.run_sync(TrainingService.submit_ensemble_train, req, background_tasks)



@router.get("/models/train/{job_id}")
async def train_status(job_id: str):
    """查询训练任务状态 (内存 → Redis 回退)"""
    return TrainingService.train_status(job_id)


@router.get("/optuna-params")
async def list_optuna_params():
    """列出所有已保存的 Optuna 搜索结果"""
    return TrainingService.list_optuna_params()


@router.post("/models/optuna-search")
async def optuna_search(req: OptunaSearchRequest, background_tasks: BackgroundTasks):
    """Optuna 超参自动搜索 (异步). search_dual=True 依次搜LGB+XGB后融合重训."""
    return await anyio.to_thread.run_sync(TrainingService.submit_optuna, req, background_tasks)


@router.post("/models/rolling-train")
async def rolling_train(req: RollingTrainRequest, background_tasks: BackgroundTasks):
    """滚动时序训练 (异步): 季度 anchor 滚动, 每轮独立 MLflow experiment"""
    return await anyio.to_thread.run_sync(TrainingService.submit_rolling_train, req, background_tasks)


@router.get("/trained-models")
async def list_trained_models():
    """列出所有已训练的模型（扫描 MLflow 实验）"""
    return await anyio.to_thread.run_sync(TrainingService.list_trained_models)


@router.get("/training/results")
async def list_training_results(
    factor_set: str = None,
    instruments: str = None,
    limit: int = 50,
    sort_by: str = "created_at",
):
    """查询历史训练结果 (从 MySQL training_results 表)"""
    return TrainingService.list_training_results(factor_set, instruments, limit, sort_by)


@router.delete("/trained-models/{experiment_id}/{run_id}")
async def delete_trained_model(experiment_id: str, run_id: str):
    """删除指定训练模型"""
    return TrainingService.delete_trained_model(experiment_id, run_id)


@router.put("/trained-models/{run_id}/disable")
async def disable_model(run_id: str, req: ModelToggleRequest):
    """停用指定模型 — 停用后所有使用该模型的地方将跳过该模型"""
    return TrainingService.disable_model(run_id, req)


@router.put("/trained-models/{run_id}/enable")
async def enable_model(run_id: str, req: ModelToggleRequest):
    """恢复已停用的模型"""
    return TrainingService.enable_model(run_id, req)
