"""Pydantic schemas for job-related API operations."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.models.job import JobStatus


class JobCreate(BaseModel):
    """Request to create a new processing job."""

    video_id: str


class JobResponse(BaseModel):
    """Job details response."""

    id: str
    video_id: str
    status: JobStatus
    progress: float = Field(ge=0, le=100)
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    """Lightweight job status response for polling."""

    id: str
    status: JobStatus
    progress: float
    current_stage: Optional[str] = None


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: list[JobResponse]
    total: int
    page: int = 1
    page_size: int = 20
