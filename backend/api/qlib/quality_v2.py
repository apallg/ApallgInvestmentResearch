"""Qlib 集成 API — 数据质量/检查/修复端点（薄接口，委托 DataQualityService）"""
import anyio
from fastapi import APIRouter, BackgroundTasks

from backend.service.data_quality import DataQualityService

router = APIRouter()


@router.get("/data/quality")
async def data_quality():
    """检查每日数据完整性 — 按股票统计覆盖范围"""
    return await anyio.to_thread.run_sync(DataQualityService.data_quality)


@router.get("/data/validate-qlib")
async def validate_qlib_data():
    """验证 Qlib 二进制数据是否正常（防止标签全0等问题）"""
    return DataQualityService.validate_qlib_data()


@router.get("/data/gaps")
async def check_data_gaps(lookback_days: int = 5, limit: int = 200):
    """检测近期有数据缺口的股票（有某些交易日但缺中间某天）"""
    return DataQualityService.check_data_gaps(lookback_days, limit)


@router.get("/data/check")
async def data_check():
    """综合数据检查：缺口 + 单位一致性 + 覆盖度"""
    return DataQualityService.data_check()


@router.post("/data/fix-units")
async def fix_unit_issues(background_tasks: BackgroundTasks):
    """检测并修复单位异常的股票（删旧数据 → Tencent重采）"""
    return DataQualityService.start_fix_units(background_tasks)
