"""API routers."""

from backend.routers.videos import router as videos_router
from backend.routers.pipeline import router as pipeline_router
from backend.routers.analysis import router as analysis_router

__all__ = ["videos_router", "pipeline_router", "analysis_router"]
