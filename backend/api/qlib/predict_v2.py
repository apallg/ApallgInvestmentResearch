"""Qlib 集成 API — 预测 + 回测端点（薄接口，委托 PredictionService）"""
from fastapi import APIRouter

from backend.api.qlib.models import PredictRequest, BacktestRequest
from backend.service.prediction import PredictionService

router = APIRouter()


@router.post("/models/predict")
async def predict(req: PredictRequest):
    """加载已训练模型并生成预测 (异步后台执行)"""
    return PredictionService.submit_predict(req)


@router.get("/models/predict/{job_id}")
async def get_predict_result(job_id: str):
    """查询预测任务结果"""
    return PredictionService.get_predict_result(job_id)


@router.post("/models/predict-dual")
async def predict_dual(req: PredictRequest):
    """双模型等权融合预测 (异步后台执行)"""
    return PredictionService.submit_predict_dual(req)


@router.get("/predictions/{code}")
async def get_stock_predictions(code: str, days: int = 60, model_id: str = None):
    """查询某只股票的历史预测分数"""
    return PredictionService.get_stock_predictions(code, days, model_id)


@router.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    """运行 qlib 回测"""
    return PredictionService.run_backtest(req)
