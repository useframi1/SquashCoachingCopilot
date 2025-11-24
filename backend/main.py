"""
SquashCoachingCopilot Backend API

FastAPI application for video analysis using the squashcopilot pipeline.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models.database import init_db
from backend.routers import videos_router, pipeline_router, analysis_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting SquashCoachingCopilot API...")
    settings.setup_directories()
    init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down SquashCoachingCopilot API...")


app = FastAPI(
    title=settings.app_name,
    description="Backend API for squash video analysis using computer vision and machine learning.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(videos_router)
app.include_router(pipeline_router)
app.include_router(analysis_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
