"""Pipeline job management API endpoints."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.models.database import get_db
from backend.models.job import JobStatus
from backend.schemas.job import JobCreate, JobResponse, JobStatusResponse, JobListResponse
from backend.services.pipeline_service import PipelineService

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def get_pipeline_service(db: Session = Depends(get_db)) -> PipelineService:
    return PipelineService(db)


def run_pipeline_task(job_id: str, db_url: str):
    """Background task to run the pipeline."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.services.pipeline_service import PipelineService

    # Create a new database session for the background task
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        service = PipelineService(db)
        service.run_pipeline(job_id)
    finally:
        db.close()


@router.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreate,
    background_tasks: BackgroundTasks,
    service: PipelineService = Depends(get_pipeline_service),
):
    """Create a new processing job for a video."""
    try:
        job = service.create_job(request.video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Start pipeline in background
    from backend.config import settings
    background_tasks.add_task(run_pipeline_task, job.id, settings.database_url)

    return JobResponse(
        id=job.id,
        video_id=job.video_id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    video_id: Optional[str] = None,
    status: Optional[JobStatus] = None,
    page: int = 1,
    page_size: int = 20,
    service: PipelineService = Depends(get_pipeline_service),
):
    """List all jobs with optional filters."""
    jobs, total = service.list_jobs(
        video_id=video_id,
        status=status,
        page=page,
        page_size=page_size,
    )

    return JobListResponse(
        jobs=[
            JobResponse(
                id=j.id,
                video_id=j.video_id,
                status=j.status,
                progress=j.progress,
                current_stage=j.current_stage,
                error_message=j.error_message,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
            )
            for j in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    service: PipelineService = Depends(get_pipeline_service),
):
    """Get job details."""
    try:
        job = service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return JobResponse(
        id=job.id,
        video_id=job.video_id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    service: PipelineService = Depends(get_pipeline_service),
):
    """Get lightweight job status for polling."""
    try:
        job = service.get_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
    )


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    service: PipelineService = Depends(get_pipeline_service),
):
    """Cancel a pending or processing job."""
    try:
        job = service.cancel_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JobResponse(
        id=job.id,
        video_id=job.video_id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
