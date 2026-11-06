"""Qlib 集成 API — 合并 router (main.py 的 import backend.api.qlib as qlib_api 不变)"""
from fastapi import APIRouter
from backend.api.qlib.collect_v2 import router as collect_router
from backend.api.qlib.quality_v2 import router as quality_router
from backend.api.qlib.train_v2 import router as train_router
from backend.api.qlib.predict_v2 import router as predict_router


router = APIRouter()
router.include_router(collect_router)
router.include_router(quality_router)
router.include_router(train_router)
router.include_router(predict_router)

