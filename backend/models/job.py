"""Job tracking database model."""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship

from backend.models.database import Base


class JobStatus(str, PyEnum):
    """Job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """Pipeline processing job."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)

    # Status tracking
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0, nullable=False)  # 0-100
    current_stage = Column(String(100), nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    video = relationship("Video", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, status={self.status}, progress={self.progress}%)>"

    def start(self) -> None:
        """Mark job as started."""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark job as completed (only if not already cancelled)."""
        # Don't overwrite CANCELLED status
        if self.status != JobStatus.CANCELLED:
            self.status = JobStatus.COMPLETED
            self.progress = 100.0
            self.completed_at = datetime.utcnow()

    def fail(self, error_message: str) -> None:
        """Mark job as failed with error message (only if not already cancelled)."""
        # Don't overwrite CANCELLED status
        if self.status != JobStatus.CANCELLED:
            self.status = JobStatus.FAILED
            self.error_message = error_message
            self.completed_at = datetime.utcnow()

    def cancel(self) -> None:
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()

    def update_progress(self, stage: str, progress: float) -> None:
        """Update job progress."""
        self.current_stage = stage
        self.progress = min(progress, 100.0)
