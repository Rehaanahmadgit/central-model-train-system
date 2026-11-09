"""Aggregated REST router under /api/v1 [doc 05]."""

from fastapi import APIRouter

from app.api.checkpoints import router as checkpoints_router
from app.api.datasets import router as datasets_router
from app.api.licences import router as licences_router
from app.api.models import router as models_router
from app.api.projects import router as projects_router
from app.api.runs import router as runs_router
from app.api.system import router as system_router
from app.api.uploads import router as uploads_router
from app.api.verifications import router as verifications_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(projects_router)
api_router.include_router(uploads_router)
api_router.include_router(datasets_router)
api_router.include_router(verifications_router)
api_router.include_router(checkpoints_router)
api_router.include_router(licences_router)
api_router.include_router(models_router)
api_router.include_router(runs_router)
api_router.include_router(system_router)
