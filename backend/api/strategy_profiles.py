from typing import Any, Dict, List, Optional

import anyio
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field


router = APIRouter()


class CreateProfileRequest(BaseModel):
    profile_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=120)
    config: Dict[str, Any] = Field(default_factory=dict)
    parent_version: Optional[int] = Field(None, ge=1)
    snapshot_as_of: Optional[str] = None


class EvaluateProfileRequest(BaseModel):
    backtest_job_ids: List[str] = Field(..., min_length=1, max_length=20)


class PaperPeriodRequest(BaseModel):
    screening_save_id: int = Field(..., ge=1)
    decision_id: str = Field(..., min_length=1, max_length=36)


class SnapshotRequest(BaseModel):
    as_of: str
    min_history_days: int = Field(500, ge=500, le=500)


class SecurityReferenceImportRequest(BaseModel):
    as_of: str
    source_name: str = Field(..., min_length=1, max_length=64)
    source_version: str = Field(..., min_length=1, max_length=128)
    source_available_at: str
    source_evidence_sha256: str = Field(..., pattern=r"^[0-9a-fA-F]{64}$")
    references: List[Dict[str, Any]] = Field(..., min_length=1)


class AbsoluteGateFitRequest(BaseModel):
    oof_rows: List[Dict[str, Any]] = Field(..., min_length=20)
    bins: int = Field(10, ge=5, le=20)


def _actor(authorization):
    from backend.api.auth import require_admin_actor
    return require_admin_actor(authorization)


def _service_call(fn, *args, status_code=422, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as error:
        raise HTTPException(status_code, str(error)) from error


@router.get("")
async def list_profiles(limit: int = Query(100, ge=1, le=500)):
    from backend.service.strategy_profile import StrategyProfileService
    return {"data": await anyio.to_thread.run_sync(lambda: StrategyProfileService.list(limit))}


@router.get("/champion")
async def get_champion():
    from backend.service.strategy_profile import StrategyProfileService
    return {"data": await anyio.to_thread.run_sync(StrategyProfileService.champion)}


@router.post("/data-snapshots")
async def create_data_snapshot(body: SnapshotRequest):
    from backend.service.strategy_profile import StrategyProfileService
    result = await anyio.to_thread.run_sync(
        lambda: _service_call(StrategyProfileService.build_data_snapshot, body.as_of, body.min_history_days)
    )
    return {"data": result}


@router.post("/security-reference-snapshots")
async def import_security_reference_snapshot(body: SecurityReferenceImportRequest, authorization: str = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    _actor(authorization)
    result = await anyio.to_thread.run_sync(
        lambda: _service_call(
            StrategyProfileService.import_security_references,
            body.as_of, body.references, body.source_name, body.source_version,
            body.source_available_at, body.source_evidence_sha256,
        )
    )
    return {"data": result}


@router.post("/absolute-gate/fit")
async def fit_absolute_gate_model(body: AbsoluteGateFitRequest, authorization: Optional[str] = Header(None)):
    _actor(authorization)
    from core.domain.screening.absolute_gate import fit_absolute_gate
    result = await anyio.to_thread.run_sync(
        lambda: _service_call(fit_absolute_gate, body.oof_rows, body.bins)
    )
    return {"data": result}


@router.get("/{profile_id}/{version}")
async def get_profile(profile_id: str, version: int):
    from backend.service.strategy_profile import StrategyProfileService
    result = await anyio.to_thread.run_sync(lambda: StrategyProfileService.get(profile_id, version))
    if not result:
        raise HTTPException(404, "策略版本不存在")
    return {"data": result}


@router.post("")
async def create_profile(body: CreateProfileRequest, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.create,
        body.profile_id, body.name, body.config, body.parent_version, body.snapshot_as_of, actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/validate")
async def validate_profile(profile_id: str, version: int, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.validate, profile_id, version, actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/evaluate")
async def evaluate_profile(profile_id: str, version: int, body: EvaluateProfileRequest, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.evaluate, profile_id, version, body.backtest_job_ids, actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/paper-start")
async def start_paper(profile_id: str, version: int, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.start_paper, profile_id, version, actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/paper-periods")
async def record_paper_period(profile_id: str, version: int, body: PaperPeriodRequest, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.record_paper_period, profile_id, version, body.model_dump(), actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/paper-sync")
async def sync_paper_periods(profile_id: str, version: int, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.sync_paper_periods, profile_id, version, actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/activate")
async def activate_profile(profile_id: str, version: int, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.activate, profile_id, version, actor,
    ))
    if not result.get("activated"):
        raise HTTPException(409, detail={"message": "生产准入门槛未满足", **result})
    return {"data": result}


@router.post("/{profile_id}/{version}/rollback")
async def rollback_profile(profile_id: str, version: int, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.rollback, profile_id, version, actor,
    ))
    return {"data": result}


@router.post("/{profile_id}/{version}/retire")
async def retire_profile(profile_id: str, version: int, authorization: Optional[str] = Header(None)):
    from backend.service.strategy_profile import StrategyProfileService
    actor = _actor(authorization)
    result = await anyio.to_thread.run_sync(lambda: _service_call(
        StrategyProfileService.retire, profile_id, version, actor,
    ))
    return {"data": result}
